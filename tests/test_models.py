"""Unit tests model_utils functions."""
from remedi import models

import pytest
import torch


def assert_equals(actual, expected, path="actual"):
    """Simple implementation of torch-friendly deep equality."""
    assert type(actual) is type(expected), path
    if isinstance(actual, torch.Tensor):
        assert actual.float().allclose(expected.float()), path
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
    "value,device,dtype",
    (
        ("x", None, None),
        (torch.tensor([1, 2, 3]), None, None),
        (torch.tensor([1, 2, 3]), None, torch.float),
        ({"foo": "bar"}, "cpu", None),
        ([torch.Tensor([1, 2, 3])], "cpu", None),
        ({"foo": torch.tensor([1, 2, 3])}, "cpu", None),
        ({"foo": [torch.tensor([1, 2, 3])]}, "cpu", None),
        ({"foo": [torch.tensor([1, 2, 3])]}, "cpu", torch.float),
    ),
)
def test_map_to_preserves_values(value, device, dtype):
    """Test map_location returns correct value."""
    actual = models.map_to(value, device=device, dtype=dtype)
    assert_equals(actual, value)
