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


def _maybe_batch(text: str | StrSequence) -> StrSequence:
    """Batch the text if it is not already batched."""
    if isinstance(text, str):
        return [text]
    return text


def _as_fp32(data: dict) -> dict:
    """Cast all top-level float tensor values to float32."""
    return {
        key: value.float()
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
        else value
        for key, value in data.items()
    }


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
    occurrence: int = 0,
) -> torch.Tensor:
    """Return shape (batch_size, 2) tensor of token ranges for (str, substr) pairs."""
    strings = _maybe_batch(strings)
    substrings = _maybe_batch(substrings)
    if len(strings) != len(substrings):
        raise ValueError(
            f"got {len(strings)} strings but only {len(substrings)} substrings"
        )

    return torch.tensor(
        [
            tokenizer_utils.find_token_range(
                string, substring, offset_mapping=offset_mapping, occurrence=occurrence
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
    words = _maybe_batch(words)
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
    return_entity_hiddens: bool = False,
    return_context_hiddens: bool = False,
    return_token_ranges: bool = True,
    return_target_token_ids: bool = True,
    return_average_entity_hiddens: bool = True,
    return_average_attribute_hiddens: bool = True,
    fp32: bool = False,
) -> dict:
    """Precompute everything the editor model needs to run from the batch."""
    mt.model.to(device)

    # Pull out expected values.
    entities = _maybe_batch(batch["entity"])
    prompts = _maybe_batch(batch["prompt"])
    contexts = _maybe_batch(batch["context"])
    attributes = _maybe_batch(batch["attribute"])

    precomputed: dict = {}

    # Precompute inputs.
    context_inputs, context_offset_mapping = None, None
    if (
        return_context_hiddens
        or return_average_attribute_hiddens
        or return_token_ranges
    ):
        context_inputs, context_offset_mapping = inputs_from_batch(
            mt, contexts, device=device
        )

    # Precompute context representations if needed.
    context_hiddens_by_layer = None
    if return_context_hiddens or return_average_attribute_hiddens:
        assert context_inputs is not None
        context_hiddens_by_layer = hiddens_from_batch(
            mt, context_inputs, layers=layers, device=device
        )
    if return_context_hiddens:
        assert context_hiddens_by_layer is not None
        for layer, hiddens in context_hiddens_by_layer.items():
            precomputed[f"context.hiddens.{layer}"] = hiddens

    # Precompute entity representations if needed.
    entity_inputs = None
    entity_hiddens_by_layer = None
    if return_entity_hiddens or return_average_entity_hiddens:
        entity_inputs, _ = inputs_from_batch(mt, entities, device=device)
        entity_hiddens_by_layer = hiddens_from_batch(mt, entity_inputs)
        for layer, hiddens in entity_hiddens_by_layer.items():
            precomputed[f"entity.hiddens.{layer}"] = hiddens

    # Precompute token ranges if needed.
    attr_ijs = None
    if return_token_ranges or return_average_attribute_hiddens:
        assert context_offset_mapping is not None
        _, prompt_offset_mapping = inputs_from_batch(mt, prompts)
        precomputed["prompt.token_range.entity"] = token_ranges_from_batch(
            prompts, entities, prompt_offset_mapping
        )
        precomputed["context.token_range.entity"] = token_ranges_from_batch(
            contexts, entities, context_offset_mapping
        )
        precomputed[
            "context.token_range.attribute"
        ] = attr_ijs = token_ranges_from_batch(
            contexts, attributes, context_offset_mapping
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

    # Precompute average entity representation if needed.
    if return_average_entity_hiddens:
        assert entity_hiddens_by_layer is not None
        assert entity_inputs is not None
        for layer, hiddens in entity_hiddens_by_layer.items():
            counts = entity_inputs.attention_mask.sum(dim=-1, keepdim=-1)
            average = hiddens.sum(dim=1) / counts
            key = f"entity.hiddens.{layer}.average"
            precomputed[key] = average

    # Precompute average attr representation if needed.
    if return_average_attribute_hiddens:
        assert context_hiddens_by_layer is not None
        assert attr_ijs is not None
        for layer, hiddens in context_hiddens_by_layer.items():
            key = f"context.hiddens.{layer}.attribute"
            precomputed[key] = average_hiddens_from_batch(hiddens, attr_ijs)

    if fp32:
        precomputed = _as_fp32(precomputed)

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


def entity_deltas_from_batch(
    mt: model_utils.ModelAndTokenizer,
    batch: dataset_utils.ContextMediationInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    return_hiddens: bool = False,
    return_token_ranges: bool = True,
    return_deltas: bool = True,
    fp32: bool = False,
) -> dict:
    """Compute in-context entity delta for the batch.

    Args:
        mt: Model and tokenizer.
        batch: Context mediation-style batch.
        layers: Layers to compute deltas in. Defaults to all.
        device: Send model and inputs to this device.
        return_hiddens: Return hidden states for contextualized prompt.
        return_token_ranges: Return entity token ranges for contextualied prompt.
        return_deltas: Return entity deltas for each layer.
        fp32: Force cast each tensor to fp32.

    Returns:
        Deltas and related precomputed values.

    """
    entities = _maybe_batch(batch["entity"])
    prompts = _maybe_batch(batch["prompt"])
    contexts = _maybe_batch(batch["context"])

    precomputed: dict = {}

    precomputed["prompt_in_context"] = prompts_in_context = [
        f"{context.rstrip('.')}. {prompt}" for context, prompt in zip(prompts, contexts)
    ]

    inputs = None
    first_entity_token_ranges = None
    last_entity_token_ranges = None
    if return_hiddens or return_token_ranges or return_deltas:
        inputs, offset_mapping = inputs_from_batch(
            mt, prompts_in_context, device=device
        )
        first_entity_token_ranges = token_ranges_from_batch(
            inputs, entities, offset_mapping
        )
        last_entity_token_ranges = token_ranges_from_batch(
            inputs, entities, offset_mapping, occurrence=1
        )
        if return_token_ranges:
            for position, token_ranges in (
                ("first", first_entity_token_ranges),
                ("last", last_entity_token_ranges),
            ):
                key = f"prompt_in_context.entity.{position}.token_range"
                precomputed[key] = token_ranges

    hiddens_by_layer = None
    if return_hiddens or return_deltas:
        assert inputs is not None
        hiddens_by_layer = hiddens_from_batch(mt, inputs, layers=layers, device=device)
        if return_hiddens:
            for layer, hiddens in hiddens_by_layer.items():
                key = f"prompt_in_context.hiddens.{layer}"
                precomputed[key] = hiddens

    if return_deltas:
        assert hiddens_by_layer is not None
        assert first_entity_token_ranges is not None
        assert last_entity_token_ranges is not None
        for layer, hiddens in hiddens_by_layer.items():
            first_entity_hiddens = average_hiddens_from_batch(
                hiddens, first_entity_token_ranges
            )
            last_entity_hiddens = average_hiddens_from_batch(
                hiddens, last_entity_token_ranges
            )
            delta = last_entity_hiddens - first_entity_hiddens

            key = f"prompt_in_context.delta.{layer}"
            precomputed[key] = delta

    if fp32:
        precomputed = _as_fp32(precomputed)

    return precomputed


def entity_deltas_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
    **kwargs: Any,
) -> Dataset:
    return dataset.map(
        partial(
            entity_deltas_from_batch,
            mt,
            layers=layers,
            device=device,
            fp32=True,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc="precompute entity deltas",
    )


def has_entity_deltas(batch: dict) -> bool:
    """Return True if the batch already has precomputed entity deltas."""
    return "prompt_in_context" in batch
