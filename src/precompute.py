"""Logic for getting and mucking with model hidden representations."""
from functools import partial
from typing import Optional, Sequence

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


def inputs_and_offset_mapping_from_batch(
    mt: model_utils.ModelAndTokenizer, text: str | StrSequence
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
    return inputs, offset_mapping


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
        inputs, _ = inputs_and_offset_mapping_from_batch(mt, inputs)
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
) -> list[list[int]]:
    """Compute token ranges."""
    if isinstance(strings, str):
        strings = [strings]
    if isinstance(substrings, str):
        substrings = [substrings]
    if len(strings) != len(substrings):
        raise ValueError(
            f"got {len(strings)} strings but only {len(substrings)} substrings"
        )

    return [
        list(
            tokenizer_utils.find_token_range(
                string, substring, offset_mapping=offset_mapping
            )
        )
        for string, substring, offset_mapping in zip(
            strings, substrings, offsets_mapping
        )
    ]


def first_token_ids_from_batch(
    mt: model_utils.ModelAndTokenizer | Tokenizer, words: str | StrSequence
) -> list[int]:
    """Return first token ID for each word."""
    if isinstance(words, str):
        words = [words]
    tokenizer = model_utils.unwrap_tokenizer(mt)
    # TODO(evandez): Centralize this spacing nonsense.
    token_ids = tokenizer([" " + word for word in words])
    return [ti[0] for ti in token_ids.input_ids]


def average_hiddens_from_batch(
    hiddens: torch.Tensor, ranges: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Compute average hidden rep in given token ranges.

    Args:
        hiddens: Should have shape (batch_size, sequence_length, hidden_size)
        ranges: Token ranges.

    Returns:
        Shape (batch_size, hidden_size) tensor containing average hiddens.

    """
    averages = []
    for bi, (ti, tj) in enumerate(ranges):
        average = hiddens[bi, ti:tj].mean(dim=0)
        averages.append(average)
    return torch.stack(averages)


def editor_inputs_from_batch(
    mt: model_utils.ModelAndTokenizer,
    batch: dataset_utils.ContextMediationInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
) -> dict:
    """Precompute everything the editor model needs to run from the batch."""
    mt.model.to(device)

    # Pull out expected values.
    entities = batch["entity"]
    prompts = batch["prompt"]
    contexts = batch["context"]
    attributes = batch["attribute"]
    target_mediated = batch["target_mediated"]
    target_unmediated = batch["target_unmediated"]

    # Precompute inputs.
    inputs_contexts = mt.tokenizer(
        contexts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    ).to(device)
    inputs_contexts_offset_mapping = inputs_contexts.pop("offset_mapping")

    inputs_prompts = mt.tokenizer(
        prompts,
        padding="longest",
        truncation=True,
        return_offsets_mapping=True,
    )

    # Precompute context representations.
    hiddens_by_layer = hiddens_from_batch(mt, inputs_contexts, layers=layers)
    precomputed: dict = {
        f"context.hiddens.{layer}": hiddens
        for layer, hiddens in hiddens_by_layer.items()
    }

    # Precomptue token ranges.
    precomputed["prompt.token_range.entity"] = token_ranges_from_batch(
        prompts, entities, inputs_prompts.offset_mapping
    )
    precomputed["context.token_range.entity"] = token_ranges_from_batch(
        contexts, entities, inputs_contexts_offset_mapping
    )
    precomputed["context.token_range.attribute"] = attr_ijs = token_ranges_from_batch(
        contexts, attributes, inputs_contexts_offset_mapping
    )

    # Precompute token IDs.
    precomputed["target_mediated.token_id"] = first_token_ids_from_batch(
        mt, target_mediated
    )
    precomputed["target_unmediated.token_id"] = first_token_ids_from_batch(
        mt, target_unmediated
    )

    # Precompute average attr representation.
    for layer, hiddens in hiddens_by_layer.items():
        key = f"context.hiddens.{layer}.attribute"
        precomputed[key] = average_hiddens_from_batch(hiddens, attr_ijs)

    return precomputed


def editor_inputs_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
) -> Dataset:
    """Precompute everything the editor model needs to train and run."""
    return dataset.map(
        partial(editor_inputs_from_batch, mt, layers=layers, device=device),
        batched=True,
        batch_size=batch_size,
        desc="precompute editor inputs",
    )


def has_editor_inputs(batch: dict) -> bool:
    """Determine if editor inputs already precomputed."""
    return "prompt.token_range.entity" in batch  # Check for just one flag entry.
