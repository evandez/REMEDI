"""Functions for loading and interacting with pretrained language models.

This module is designed to house all the annoying branching logic
that comes with supporting analysis of many slightly different model
implementations.
"""
from typing import Any, NamedTuple, Optional, Sequence

from src.utils.typing import Device, Model, Tokenizer

import torch
import transformers

SUPPORTED_MODELS = ("gpt2", "gpt2-xl")


class ModelAndTokenizer(NamedTuple):
    """A pretrained model and its tokenizer."""

    model: Model
    tokenizer: Tokenizer


def load_model(name: str, device: Optional[Device] = None) -> ModelAndTokenizer:
    """Load the model given its string name.

    Args:
        name: Name of the model.
        device: If set, send model to this device. Defaults to CPU.

    Returns:
        ModelAndTokenizer: Loaded model and its tokenizer.

    """
    model = transformers.AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return ModelAndTokenizer(model, tokenizer)


def _unwrap_model(value: Model | ModelAndTokenizer) -> Model:
    """Unwrap the model if necessary."""
    if isinstance(value, ModelAndTokenizer):
        return value.model
    return value


def determine_layers(model: ModelAndTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = _unwrap_model(model)
    if isinstance(model, transformers.GPT2LMHeadModel):
        return tuple(range(model.config.n_layer))
    else:
        raise ValueError(f"unknown model type: {model.__class__.__name__}")


def determine_layer_paths(
    model: ModelAndTokenizer | Model, layers: Optional[Sequence[int]] = None
) -> tuple[str, ...]:
    """Determine the absolute paths to the given layers in the model.

    Args:
        model: The model.
        layers: The specific layer (numbers) to look at. Defaults to all of them.

    Returns:
        The paths to each layer.

    """
    model = _unwrap_model(model)
    if layers is None:
        layers = determine_layers(model)
    if isinstance(model, transformers.GPT2LMHeadModel):
        return tuple(f"transformer.h.{layer}" for layer in layers)
    else:
        raise ValueError(f"unknown model type: {model.__class__.__name__}")


def determine_mlp_fan_out_layer_paths(
    model: ModelAndTokenizer | Model, layers: Optional[Sequence[int]] = None
) -> tuple[str, ...]:
    """Determine MLP fan out layer paths."""
    model = _unwrap_model(model)
    layer_paths = determine_layer_paths(model, layers=layers)
    if isinstance(model, transformers.GPT2LMHeadModel):
        return tuple(f"{layer_path}.mlp.c_fc" for layer_path in layer_paths)
    else:
        raise ValueError(f"unknown model type: {model.__class__.__name__}")


def determine_mlp_fan_in_layer_paths(
    model: ModelAndTokenizer | Model, layers: Optional[Sequence[int]] = None
) -> tuple[str, ...]:
    """Determine MLP fan in layer paths."""
    model = _unwrap_model(model)
    layer_paths = determine_layer_paths(model, layers=layers)
    if isinstance(model, transformers.GPT2LMHeadModel):
        return tuple(f"{layer_path}.mlp.c_proj" for layer_path in layer_paths)
    else:
        raise ValueError(f"unknown model type: {model.__class__.__name__}")


def determine_hidden_size(model: ModelAndTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = _unwrap_model(model)
    return model.config.hidden_size


def map_location(orig: Any, device: Device | None) -> Any:
    """Map all tensors in the given value to the device.
    Args:
        orig: Any sequence of or mapping to tensors, or just a tensor.
        device: Device to send to.
    Returns:
        Same value, but with all tensors moved to the device.
    """
    if device is None:
        return orig

    result = orig
    if isinstance(orig, torch.Tensor):
        result = orig.to(device)
    elif isinstance(orig, dict):
        result = {key: map_location(value, device) for key, value in orig.items()}
    elif isinstance(orig, (list, tuple)):
        result = orig.__class__(map_location(value, device) for value in orig)
    assert type(result) is type(orig)
    return result
