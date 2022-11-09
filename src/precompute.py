"""Logic for getting and mucking with model hidden representations."""
from functools import partial
from typing import Optional, Sequence

from src.utils import model_utils, tokenizer_utils
from src.utils.typing import Dataset, Device, StrSequence, Tokenizer

import torch
from baukit import nethook


def _unwrap_tokenizer(
    tokenizer: model_utils.ModelAndTokenizer | Tokenizer,
) -> Tokenizer:
    """Unwrap the tokenizer."""
    if isinstance(tokenizer, model_utils.ModelAndTokenizer):
        return tokenizer.tokenizer
    return tokenizer


def _token_ranges(
    strings: StrSequence,
    substrings: StrSequence,
    offsets_mapping: Sequence[Sequence[tuple[int, int]]],
) -> torch.Tensor:
    """Compute token ranges."""
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


def _first_token_ids(
    mt: model_utils.ModelAndTokenizer | Tokenizer, words: StrSequence
) -> list[int]:
    """Return first token ID for each word."""
    tokenizer = _unwrap_tokenizer(mt)
    token_ids = tokenizer.batch_encode_plus(words)
    return [ti[0] for ti in token_ids]


def _average_hiddens(hiddens: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Compute average hidden rep in given token ranges.

    Args:
        hiddens: Should have shape (batch_size, sequence_length, hidden_size)
        ranges: Token ranges.

    Returns:
        Shape (batch_size, hidden_size) tensor containing average hiddens.

    """
    averages = []
    for bi, (ti, tj) in enumerate(ranges.tolist()):
        average = hiddens[bi, ti:tj].mean(dim=0)
        averages.append(average)
    return torch.stack(averages)


def editor_inputs_from_batch(
    mt: model_utils.ModelAndTokenizer,
    batch: dict[str, StrSequence],
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
) -> dict:
    """Precompute everything the editor model needs to run from the batch."""
    mt.model.to(device)
    layer_paths = model_utils.determine_layer_paths(mt, layers=layers, return_dict=True)

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
    with nethook.TraceDict(mt.model, layers=layer_paths.values(), stop=True) as ret:
        mt.model(**inputs_contexts)
    hiddens_by_layer = {
        layer: ret[layer_path].output[0].cpu()
        for layer, layer_path in layer_paths.items()
    }
    precomputed = {
        f"hiddens.{layer}": hiddens for layer, hiddens in hiddens_by_layer.items()
    }

    # Precomptue token ranges.
    precomputed["prompt.token_range.entity"] = _token_ranges(
        prompts, entities, inputs_prompts.offset_mapping
    )
    precomputed["context.token_range.entity"] = _token_ranges(
        contexts, entities, inputs_contexts_offset_mapping
    )
    precomputed["context.token_range.attribute"] = attr_ijs = _token_ranges(
        contexts, attributes, inputs_contexts_offset_mapping
    )

    # Precompute token IDs.
    precomputed["target_mediated.token_id"] = _first_token_ids(mt, target_mediated)
    precomputed["target_unmediated.token_id"] = _first_token_ids(mt, target_unmediated)

    # Precompute average attr representation.
    for layer, hiddens in hiddens_by_layer.items():
        key = f"context.hiddens.{layer}.attribute"
        precomputed[key] = _average_hiddens(hiddens, attr_ijs)

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
