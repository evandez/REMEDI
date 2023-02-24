"""Unit tests for lang_utils."""
from remedi.utils import lang_utils

import pytest


@pytest.mark.parametrize(
    "word,expected",
    (
        ("mink", "a"),
        ("airplane", "an"),
        ("boy", "a"),
        ("foobarbaz", "a"),
        ("open", "an"),
    ),
)
def test_determine_article(word, expected):
    """Test determine_article produces correct article."""
    actual = lang_utils.determine_article(word)
    assert actual == expected
