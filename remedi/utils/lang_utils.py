"""Heuristics and tools for cleaning language."""
import logging
from functools import cache

from nltk.corpus import cmudict

logger = logging.getLogger(__name__)


@cache
def _cmudict() -> dict[str, list[str]]:
    """Read the nltk cmudict corpus."""
    return cmudict.dict()


def determine_article(word: str, default: str = "a") -> str:
    """Determine article that should precede text."""
    pronounciation = _cmudict().get(word, [])
    if not pronounciation:
        logger.debug(f"no pronounciation found for: {word}")
        return default
    is_vowel_sound = pronounciation[0][-1].isdigit()
    return "an" if is_vowel_sound else "a"
