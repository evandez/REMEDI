"""Unit tests model_utils functions."""
from src.utils import model_utils

import torch
import pytest


def assert_equals(actual, expected, path="actual"):
    """Simple implementation of torch-friendly deep equality."""
    assert type(actual) is type(expected), path
    if isinstance(actual, torch.Tensor):
        assert actual.allclose(expected), path
    elif isinstance(actual, dict):
        for key, value in actual.items():
            child_path = f'{path}["{key}"]'
            assert key in expected, child_path
            assert_equals(value, expected[key], path=child_path)
    elif isinstance(actual, (list, tuple)):
        for index, value in enumerate(actual):
            child_path = f"{path}[{index}]"
            assert_equals(value, expected[index], path=child_path)


@pytest.mark.parametrize(
    "value,device",
    (
        ("x", None),
        (torch.tensor([1, 2, 3]), None),
        ({"foo": "bar"}, "cpu"),
        ([torch.Tensor([1, 2, 3])], "cpu"),
        ({"foo": torch.tensor([1, 2, 3])}, "cpu"),
        ({"foo": [torch.tensor([1, 2, 3])]}, "cpu"),
    ),
)
def test_map_location_preserves_values(value, device):
    """Test map_location returns correct value."""
    actual = model_utils.map_location(value, device)
    assert_equals(actual, value)
