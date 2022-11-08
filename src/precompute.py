"""Logic for getting and mucking with model hidden representations."""
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
    hiddens = {
        str(layer): ret[layer_path].output[0].cpu()
        for layer, layer_path in layers_to_layer_path.items()
    }
    return {"hiddens": hiddens, **inputs}


def hiddens_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    columns: StrSequence,
    layers: Optional[Sequence[int]] = None,
    layer_paths: Optional[StrSequence] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
    # store_on_device: bool = False,
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
        store_on_device: If set, keep precomputed data on device. This is ideal for
            very small datasets (where hidden reps for the whole dataset can fit on
            one GPU), but otherwise should be left off, in which case data is always
            moved to CPU.

    Returns:
        Original dataset, but with additional fields of the form
            `{"precomputed": {"column_name": {"hiddens": ...}}}`.

    """
    layers_to_layer_path = _resolve_layers(mt, layers, layer_paths)

    def _device_mapped_hiddens_from_batch(batch: dict[str, StrSequence]) -> dict:
        """Wraps `hiddens_from_batch` and handles moving data to correct device."""
        hiddens = {
            column: hiddens_from_batch(
                mt, batch[column], layer_paths=layer_paths, device=device
            )
            for column in columns
        }
        # TODO(evandez): Maybe delete.
        # if not store_on_device:
        #     hiddens = model_utils.map_location(hiddens, "cpu")
        return {"precomputed": hiddens}

    # Make a nice description.
    layers_listed = ", ".join(str(l) for l in layers_to_layer_path)
    columns_listed = ", ".join(columns)
    desc = f"precompute l=[{layers_listed}] c=[{columns_listed}]"

    return dataset.map(
        _device_mapped_hiddens_from_batch,
        batch_size=batch_size,
        desc=desc,
    ).flatten()


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
        "entity_token_range_in_prompt": tokenizer_utils.find_token_range(
            prompt, entity, tokenizer
        ),
        "entity_token_range_in_prompt": tokenizer_utils.find_token_range(
            context, entity, tokenizer
        ),
        "attr_token_range_in_context": tokenizer_utils.find_token_range(
            context, attribute, tokenizer
        ),
    }


def token_ids_from_sample(
    tokenizer: model_utils.ModelAndTokenizer | Tokenizer,
    sample: dataset_utils.ContextMediationSample,
) -> dict[str, int]:
    tokenizer = _unwrap_tokenizer(tokenizer)
    target_mediated = sample["target_mediated"]
    target_unmediated = sample["target_unmediated"]
    # TODO(evan): Detect automatically if the spacing thing is needed.
    return {
        "target_mediated_id": tokenizer(" " + target_mediated).input_ids[0],
        "target_unmediated_id": tokenizer(" " + target_unmediated).input_ids[0],
    }


def editor_inputs_from_dataset(
    mt: model_utils.ModelAndTokenizer,
    dataset: Dataset,
    precompute_hiddens_batch_size: int = 64,
    precompute_tokens_batch_size: int = 512,
    **kwargs: Any,
) -> Dataset:
    """Precompute everything the editor model needs to train and run."""
    dataset = hiddens_from_dataset(
        mt, dataset, ["context"], batch_size=precompute_hiddens_batch_size, **kwargs
    )
    for name, fn in (
        ("token ranges", token_ranges_from_sample),
        ("target word tokens", token_ids_from_sample),
    ):
        dataset = dataset.map(
            lambda sample: {"precomputed": fn(mt, sample)},
            batch_size=precompute_tokens_batch_size,
            desc=f"precompute {name}",
        )
        dataset = dataset.flatten()
    return dataset
