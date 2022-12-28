"""Functions for loading and interacting with pretrained language models.

This module is designed to house all the annoying branching logic
that comes with supporting analysis of many slightly different model
implementations.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, Sequence, overload

from src.utils.typing import Device, Model, Tokenizer

import torch
import transformers

GPT_J_NAME = "EleutherAI/gpt-j-6B"
SUPPORTED_MODELS = ("gpt2-small", "gpt2", "gpt2-xl", GPT_J_NAME)
EMBEDDING_LAYER = -1


@dataclass(frozen=True)
class ModelAndTokenizer:
    """A pretrained model and its tokenizer."""

    model: Model
    tokenizer: Tokenizer

    def to_(self, device: Optional[Device]) -> None:
        """Send model to the device."""
        self.model.to(device)

    def eval_(self) -> None:
        """Set model to eval mode."""
        self.model.eval()


def unwrap_model(value: Model | ModelAndTokenizer) -> Model:
    """Unwrap the model if necessary."""
    if isinstance(value, ModelAndTokenizer):
        return value.model
    return value


def unwrap_tokenizer(tokenizer: ModelAndTokenizer | Tokenizer) -> Tokenizer:
    """Unwrap the tokenizer."""
    if isinstance(tokenizer, ModelAndTokenizer):
        return tokenizer.tokenizer
    return tokenizer


def load_model(
    name: str, device: Optional[Device] = None, fp16: bool = False
) -> ModelAndTokenizer:
    """Load the model given its string name.

    Args:
        name: Name of the model.
        device: If set, send model to this device. Defaults to CPU.
        fp16: Use half precision.

    Returns:
        ModelAndTokenizer: Loaded model and its tokenizer.

    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model: {name}")

    torch_dtype = torch.float16 if fp16 else None

    kwargs: dict = dict(torch_dtype=torch_dtype)
    if name == GPT_J_NAME:
        kwargs["low_cpu_mem_usage"] = True
        if fp16:
            kwargs["revision"] = "float16"

    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return ModelAndTokenizer(model, tokenizer)


def determine_layers(model: ModelAndTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = unwrap_model(model)
    assert isinstance(model, Model)
    return (EMBEDDING_LAYER, *range(model.config.n_layer))


@overload
def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = ...,
    *,
    return_dict: Literal[False] = ...,
) -> Sequence[str]:
    """Determine layer path for each layer."""
    ...


@overload
def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = ...,
    *,
    return_dict: Literal[True],
) -> dict[int, str]:
    """Determine mapping from layer to layer path."""
    ...


def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = None,
    *,
    return_dict: bool = False,
) -> Sequence[str] | dict[int, str]:
    """Determine the absolute paths to the given layers in the model.

    Args:
        model: The model.
        layers: The specific layer (numbers) to look at. Defaults to all of them.
        return_dict: If True, return mapping from layer to layer path,
            otherwise just return list of layer paths in same order as `layers`.

    Returns:
        Mapping from layer number to layer path.

    """
    model = unwrap_model(model)

    if layers is None:
        layers = determine_layers(model)

    assert isinstance(model, Model), type(model)

    layer_paths = {}
    for layer in layers:
        if layer == EMBEDDING_LAYER:
            layer_path = "transformer.wte"
        else:
            layer_path = f"transformer.h.{layer}"
        layer_paths[layer] = layer_path

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


def determine_hidden_size(model: ModelAndTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)
    return model.config.hidden_size


def determine_device(model: ModelAndTokenizer | Model) -> torch.device | None:
    """Determine device model is running on."""
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None


def determine_dtype(model: ModelAndTokenizer | Model) -> torch.dtype | None:
    """Determine dtype of model."""
    parameter = any_parameter(model)
    return parameter.dtype if parameter is not None else None


def any_parameter(model: ModelAndTokenizer | Model) -> torch.nn.Parameter | None:
    """Get any example parameter for the model."""
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)


@contextmanager
def set_padding_side(
    tokenizer: Tokenizer | ModelAndTokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """Temporarily set padding side for tokenizer.

    Useful for when you want to batch generate with causal LMs like GPT, as these
    require the padding to be on the left side in such settings but are much easier
    to mess around with when the padding, by default, is on the right.

    Example usage:
        mt = model_utils.load_model("gpt2-x")
        with model_utils.set_padding_side(mt, "left"):
            inputs = mt.tokenizer(...)
        mt.model.generate(**inputs)

    """
    tokenizer = unwrap_tokenizer(tokenizer)
    _padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = _padding_side


def map_to(
    orig: Any, device: Device | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Map all tensors in the given value to the device.

    Args:
        orig: Any sequence of or mapping to tensors, or just a tensor.
        device: Device to send to.

    Returns:
        Same value, but with all tensors moved to the device.

    """
    if device is None and dtype is None:
        return orig

    result = orig
    if isinstance(orig, torch.Tensor):
        result = orig.to(device=device, dtype=dtype)
    elif isinstance(orig, dict):
        result = {
            key: map_to(value, device=device, dtype=dtype)
            for key, value in orig.items()
        }
    elif isinstance(orig, (list, tuple)):
        result = orig.__class__(
            map_to(value, device=device, dtype=dtype) for value in orig
        )
    assert isinstance(result, orig.__class__), f"{type(result)}/{type(orig)}"
    return result
