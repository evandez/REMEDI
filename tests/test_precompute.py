"""Unit tests for precompute functions."""
from remedi import precompute

import pytest
import torch


@pytest.mark.parametrize(
    "token_ranges,expected",
    (
        ([[0, 1]], [[0, 1]]),
        ([[1, 3]], [[2, 3]]),
        ([[0, 1], [1, 3]], [[0, 1], [2, 3]]),
    ),
)
def test_last_token_ranges_from_batch(token_ranges, expected):
    """Test last_token_ranges_from batch makes correct ranges."""
    actual = precompute.last_token_ranges_from_batch(torch.tensor(token_ranges))
    assert actual.eq(torch.tensor(expected)).all()


@pytest.mark.parametrize(
    "token_ranges,lengths,expected",
    (
        ([[4, 5]], [6], [[-2, -1]]),
        ([[0, 3]], [5], [[-5, -2]]),
        ([[4, 5], [0, 3]], [6, 5], [[-2, -1], [-5, -2]]),
    ),
)
def test_negative_token_ranges_from_batch(token_ranges, lengths, expected):
    """Test negative_token_ranges_from_batch makes correct ranges."""
    actual = precompute.negative_token_ranges_from_batch(
        torch.tensor(token_ranges),
        torch.tensor(lengths),
    )
    assert actual.eq(torch.tensor(expected)).all()
