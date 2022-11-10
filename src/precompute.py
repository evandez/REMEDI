"""Logic for getting and mucking with model hidden representations."""
from functools import partial
from typing import Any, Optional, Sequence, cast

from src.utils import dataset_utils, model_utils, tokenizer_utils
from src.utils.typing import (
    Dataset,
    Device,
    ModelInput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)

import torch
from baukit import nethook


def inputs_from_batch(
    mt: model_utils.ModelAndTokenizer,
    text: str | StrSequence,
    device: Optional[Device] = None,
) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
    """Precompute model inputs."""
    inputs = mt.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    if device is not None:
        inputs = inputs.to(device)
    return inputs, offset_mapping


@torch.inference_mode()
def hiddens_from_batch(
    mt: model_utils.ModelAndTokenizer,
    inputs: str | StrSequence | ModelInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
) -> dict[int, torch.Tensor]:
    """Precomptue hidden reps.

    Args:
        mt: The model and tokenizer.
        inputs: The model inputs.
        layers: Layers to compute hiddens for. Defaults to all.

    Returns:
        Hidden reps mapped by layer.

    """
    mt.to_(device)
    if isinstance(inputs, str | list | tuple):
        inputs, _ = inputs_from_batch(mt, inputs)
    if device is not None:
        inputs = inputs.to(device)
    layer_paths = model_utils.determine_layer_paths(mt, layers=layers, return_dict=True)
    with nethook.TraceDict(mt.model, layers=layer_paths.values(), stop=True) as ret:
        mt.model(**inputs)
    hiddens_by_layer = {
        layer: ret[layer_path].output[0].cpu()
        for layer, layer_path in layer_paths.items()
    }
    return hiddens_by_layer


def token_ranges_from_batch(
    strings: str | StrSequence,
    substrings: str | StrSequence,
    offsets_mapping: Sequence[TokenizerOffsetMapping],
) -> torch.Tensor:
    """Return shape (batch_size, 2) tensor of token ranges for (str, substr) pairs."""
    if isinstance(strings, str):
        strings = [strings]
    if isinstance(substrings, str):
        substrings = [substrings]
    if len(strings) != len(substrings):
        raise ValueError(
            f"got {len(strings)} strings but only {len(substrings)} substrings"
        )

    return torch.tensor(
        [
            tokenizer_utils.find_token_range(
                string, substring, offset_mapping=offset_mapping
            )
            for string, substring, offset_mapping in zip(
                strings, substrings, offsets_mapping
            )
        ]
    )


def first_token_ids_from_batch(
    mt: model_utils.ModelAndTokenizer | Tokenizer, words: str | StrSequence
) -> torch.Tensor:
    """Return shape (batch_size,) int tensor with first token ID for each word."""
    if isinstance(words, str):
        words = [words]
    tokenizer = model_utils.unwrap_tokenizer(mt)
    # TODO(evandez): Centralize this spacing nonsense.
    token_ids = tokenizer([" " + word for word in words])
    return torch.tensor([ti[0] for ti in token_ids.input_ids])


def average_hiddens_from_batch(
    hiddens: torch.Tensor, ranges: Sequence[Sequence[int]] | torch.Tensor
) -> torch.Tensor:
    """Compute average hidden rep in given token ranges.

    Args:
        hiddens: Should have shape (batch_size, sequence_length, hidden_size)
        ranges: Token ranges.

    Returns:
        Shape (batch_size, hidden_size) tensor containing average hiddens.

    """
    if not isinstance(ranges, torch.Tensor):
        ranges = torch.tensor(ranges)
    if ranges.shape != (hiddens.shape[0], 2):
        raise ValueError(f"unexpected ranges shape: {ranges.shape}")

    averages = []
    for bi, (ti, tj) in enumerate(ranges.tolist()):
        average = hiddens[bi, ti:tj].mean(dim=0)
        averages.append(average)

    return torch.stack(averages)


def editor_inputs_from_batch(
    mt: model_utils.ModelAndTokenizer,
    batch: dataset_utils.ContextMediationInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    return_hiddens: bool = False,
    return_token_ranges: bool = True,
    return_target_token_ids: bool = True,
    return_average_attribute_hiddens: bool = True,
    fp32: bool = False,
) -> dict:
    """Precompute everything the editor model needs to run from the batch."""
    mt.model.to(device)

    # Pull out expected values.
    entities = batch["entity"]
    prompts = batch["prompt"]
    contexts = batch["context"]
    attributes = batch["attribute"]

    precomputed: dict = {}

    # Precompute inputs.
    inputs_contexts, offset_mapping_contexts = None, None
    if return_hiddens or return_average_attribute_hiddens or return_token_ranges:
        inputs_contexts, offset_mapping_contexts = inputs_from_batch(
            mt, contexts, device=device
        )

    # Precompute context representations if needed.
    hiddens_by_layer = None
    if return_hiddens or return_average_attribute_hiddens:
        assert inputs_contexts is not None
        hiddens_by_layer = hiddens_from_batch(
            mt, inputs_contexts, layers=layers, device=device
        )
    if return_hiddens:
        assert hiddens_by_layer is not None
        for layer, hiddens in hiddens_by_layer.items():
            precomputed[f"context.hiddens.{layer}"] = hiddens

    # Precompute token ranges if needed.
    attr_ijs = None
    if return_token_ranges:
        assert offset_mapping_contexts is not None
        _, offset_mapping_prompts = inputs_from_batch(mt, prompts)
        precomputed["prompt.token_range.entity"] = token_ranges_from_batch(
            prompts, entities, offset_mapping_prompts
        )
        precomputed["context.token_range.entity"] = token_ranges_from_batch(
            contexts, entities, offset_mapping_contexts
        )
        precomputed[
            "context.token_range.attribute"
        ] = attr_ijs = token_ranges_from_batch(
            contexts, attributes, offset_mapping_contexts
        )

    # Precompute token IDs if needed.
    if return_target_token_ids:
        for target_key in ("target_mediated", "target_unmediated"):
            target = batch.get(target_key)
            if target is None:
                raise ValueError(
                    f'return_target_token_ids=True, but "{target_key}" not set on batch'
                )
            precomputed[f"{target_key}.token_id"] = first_token_ids_from_batch(
                mt, cast(str, target)
            )

    # Precompute average attr representation if needed.
    if return_average_attribute_hiddens:
        assert hiddens_by_layer is not None
        assert attr_ijs is not None
        for layer, hiddens in hiddens_by_layer.items():
            key = f"context.hiddens.{layer}.attribute"
            precomputed[key] = average_hiddens_from_batch(hiddens, attr_ijs)

    if fp32:
        precomputed = {
            key: value.float()
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
            else value
            for key, value in precomputed.items()
        }

    return precomputed


def editor_inputs_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
    **kwargs: Any,
) -> Dataset:
    """Precompute everything the editor model needs to train and run."""
    if "fp32" in kwargs:
        raise ValueError("cannot set fp32= because arrow datasets only support fp32")
    return dataset.map(
        partial(
            editor_inputs_from_batch,
            mt,
            layers=layers,
            device=device,
            fp32=True,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc="precompute editor inputs",
    )


def has_editor_inputs(batch: dict) -> bool:
    """Determine if editor inputs already precomputed."""
    return "prompt.token_range.entity" in batch  # Check for just one flag entry.
