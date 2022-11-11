"""Functions for loading and interacting with pretrained language models.

This module is designed to house all the annoying branching logic
that comes with supporting analysis of many slightly different model
implementations.
"""
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, overload

from src.utils.typing import Device, Model, Tokenizer

import torch
import transformers

SUPPORTED_MODELS = ("gpt2", "gpt2-xl")


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
    torch_dtype = torch.float16 if fp16 else None
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch_dtype
    )
    model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return ModelAndTokenizer(model, tokenizer)


def determine_layers(model: ModelAndTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = unwrap_model(model)
    if isinstance(model, transformers.GPT2LMHeadModel):
        return tuple(range(model.config.n_layer))
    else:
        raise ValueError(f"unknown model type: {model.__class__.__name__}")


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

    if isinstance(model, transformers.GPT2LMHeadModel):
        layer_paths = {layer: f"transformer.h.{layer}" for layer in layers}
    else:
        raise ValueError(f"unknown model type: {model.__class__.__name__}")

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


def determine_hidden_size(model: ModelAndTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)
    return model.config.hidden_size


def determine_device(model: ModelAndTokenizer | Model) -> Optional[torch.device]:
    """Determine device model is running on."""
    model = unwrap_model(model)
    parameter = next(iter(model.parameters()), None)
    return parameter.device if parameter is not None else None


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
