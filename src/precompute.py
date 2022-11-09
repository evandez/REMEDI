"""Logic for getting and mucking with model hidden representations."""
from functools import partial
from typing import Any, Optional, Sequence

from src.utils import dataset_utils, model_utils, tokenizer_utils
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


def _resolve_layers(
    mt: model_utils.ModelAndTokenizer,
    layers: Optional[Sequence[int]],
    layer_paths: Optional[StrSequence],
) -> dict[int, str]:
    """Resolve layers and layer paths."""
    if layers is not None and layer_paths is not None:
        raise ValueError("cannot set both `layers` and `layer_paths`")

    # If no layers specified or only layer numbers, get layer paths.
    if layer_paths is None:
        return model_utils.determine_layer_paths(mt, layers=layers)

    # Otherwise, need to determine layer number from layer path.
    l_to_lp = model_utils.determine_layer_paths(mt)
    return {l: lp for l, lp in l_to_lp.items() if lp in layer_paths}


def _get_first_token_ids(
    tokenizer: model_utils.ModelAndTokenizer | Tokenizer, batch: StrSequence
) -> Sequence[int]:
    """Get IDs of first token in batch."""
    tokenizer = _unwrap_tokenizer(tokenizer)
    return tokenizer(batch, padding=True, return_tensors="pt").input_ids[:, 0].tolist()


@torch.inference_mode()
def hiddens_from_batch(
    mt: model_utils.ModelAndTokenizer,
    batch: StrSequence,
    layers: Optional[Sequence[int]] = None,
    layer_paths: Optional[StrSequence] = None,
    device: Optional[Device] = None,
) -> dict:
    """Compute hidden representations for the batch."""
    layers_to_layer_path = _resolve_layers(mt, layers, layer_paths)
    layer_paths = tuple(layers_to_layer_path.values())

    mt.model.to(device)
    inputs = mt.tokenizer(
        batch, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)
    with nethook.TraceDict(mt.model, layers=layer_paths, stop=True) as ret:
        mt.model(**inputs)
    precomputed = {
        f"hiddens.{layer}": ret[layer_path].output[0].cpu()
        for layer, layer_path in layers_to_layer_path.items()
    }
    precomputed.update({f"inputs.{key}": value for key, value in inputs.items()})
    return precomputed


def hiddens_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    columns: StrSequence,
    layers: Optional[Sequence[int]] = None,
    layer_paths: Optional[StrSequence] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
) -> Dataset:
    """Precompute hidden representations for all samples in the dataset.

    Args:
        mt: The model and tokenizer.
        dataset: Dataset to compute hiddens for.
        columns: Columns to compute hiddens for.
        layers: Layer numbers to compute hiddens for. Defaults to all.
        layer_paths: Layer paths to compute hiddens for. Cannot be set with `layers`.
            Defaults to all layers.
        device: Send model and inputs to this device. Defaults to cpu.
        batch_size: Number of inputs to feed model at once. Defaults to 64.
            Note the return dataset will maintain all necessary padding tokens, meaning
            if you plan to batch if again, you will have to use the same batch size
            and not shuffle it for PyTorch data loaders to work properly.

    Returns:
        Original dataset, but with additional fields of the form
            `{"precomputed": {"column_name": {"hiddens": ...}}}`.

    """
    layers_to_layer_path = _resolve_layers(mt, layers, layer_paths)
    layers_listed = ", ".join(str(l) for l in layers_to_layer_path)
    columns_listed = ", ".join(columns)
    desc = f"precompute l=[{layers_listed}] c=[{columns_listed}]"

    def _device_mapped_hiddens_from_batch(batch: dict[str, StrSequence]) -> dict:
        """Wraps `hiddens_from_batch` and handles moving data to correct device."""
        precomputed = {}
        for column in columns:
            hiddens = hiddens_from_batch(
                mt,
                batch[column],
                layer_paths=tuple(layers_to_layer_path.values()),
                device=device,
            )
            precomputed.update(
                {f"{column}.{key}": value for key, value in hiddens.items()}
            )
        return precomputed

    return dataset.map(
        _device_mapped_hiddens_from_batch,
        batched=True,
        batch_size=batch_size,
        desc=desc,
    )


def token_ranges_from_sample(
    tokenizer: model_utils.ModelAndTokenizer | Tokenizer,
    sample: dataset_utils.ContextMediationSample,
) -> dict[str, tuple[int, int]]:
    """Compute important token ranges from sample.

    Specifically, this computes token ranges for:
    - entity in prompt
    - entity in context
    - attribute in context
    """
    tokenizer = _unwrap_tokenizer(tokenizer)
    entity = sample["entity"]
    prompt = sample["prompt"]
    context = sample["context"]
    attribute = sample["attribute"]
    return {
        "prompt.token_range.entity": tokenizer_utils.find_token_range(
            prompt, entity, tokenizer
        ),
        "context.token_range.entity": tokenizer_utils.find_token_range(
            context, entity, tokenizer
        ),
        "context.token_range.attribute": tokenizer_utils.find_token_range(
            context, attribute, tokenizer
        ),
    }


def token_ids_from_batch(
    tokenizer: model_utils.ModelAndTokenizer | Tokenizer,
    sample: dict[str, StrSequence],
) -> dict[str, Sequence[int]]:
    token_ids = {}
    for key in ("target_mediated", "target_unmediated"):
        words = [" " + word for word in sample[key]]
        token_ids[f"{key}.token_id"] = _get_first_token_ids(tokenizer, words)
    return token_ids


def editor_inputs_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    precompute_hiddens_batch_size: int = 64,
    precompute_token_ids_batch_size: int = 1024,
    **kwargs: Any,
) -> Dataset:
    """Precompute everything the editor model needs to train and run."""
    dataset = hiddens_from_dataset(
        mt, dataset, ["context"], batch_size=precompute_hiddens_batch_size, **kwargs
    )
    dataset = dataset.map(
        partial(token_ranges_from_sample, mt), desc="precompute token ranges"
    )
    dataset = dataset.map(
        partial(token_ids_from_batch, mt),
        batched=True,
        batch_size=precompute_token_ids_batch_size,
        desc="precompute token ids",
    )
    return dataset
