"""Metrics used for evaluating the editors.

Lots of this code is taken from the ROME CounterFact evaluation, for which the source
code can be found: https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_counterfact.py
"""
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from src.utils.typing import ArrayLike, StrSequence

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


def _validate_same_length(**kwargs: Sequence) -> None:
    """Validate all batch sizes are the same."""
    lengths = {key: len(seq) for key, seq in kwargs.items()}
    if len(lengths) > 1:
        message = f"inconsistent batch sizes:" + "\n"
        message += "\n".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(message)


def _n_gram_counts(text: str, n: int) -> dict[tuple[str, ...], int]:
    """Return the n-gram counts for the text."""
    tokens = nltk.word_tokenizer(text)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def _n_gram_entropy(text: str, n: int) -> float:
    """Return entropy of n-gram distribution in text."""
    counts = _n_gram_counts(text, n)
    dist = np.array([count for _, count in counts.items()])
    dist /= dist.sum()
    entropy = np.sum(-dist * np.log(dist) / np.log(2))
    return entropy.item()


def _aggregate_n_gram_entropy(
    texts: StrSequence,
    ns: Sequence[int] = (2, 3),
    weights: Sequence[float] = (2 / 3, 4 / 3),
) -> float:
    """Return average entropy across different n-gram distributions and texts."""
    _validate_same_length(ns=ns, weights=weights)
    entropies = []
    for text in texts:
        entropies_by_n = np.array([_n_gram_entropy(text, n) for n in ns])
        entropy = np.mean(entropies_by_n * np.array(weights))
        entropies.append(entropy)
    return np.array(entropies).mean()


def _tfidf_similarity(
    source: str | StrSequence, reference: str | StrSequence, vectorizer: TfidfVectorizer
) -> float:
    """Return TfIdf similarity between the texts."""
    if isinstance(source, str):
        source = [source]
    if isinstance(reference, str):
        reference = [reference]
    sv, rv = vectorizer.transform([" ".join(source), " ".join(reference)]).A
    return np.dot(sv, rv) / np.linalg.norm(sv) / np.linalg.norm(rv)


@dataclass(frozen=True)
class Metric:
    """An aggregate metric."""

    mean: float
    std: float

    @staticmethod
    def aggregate(values: ArrayLike) -> "Metric":
        return Metric(np.mean(values), np.std(values))


@dataclass(frozen=True)
class EfficacyMetrics:
    """Efficacy metrics."""

    score: Metric
    magnitude: Metric


def efficacy(samples: Sequence[dict], key: str = "prompt") -> EfficacyMetrics:
    """Compute efficacy on metrics."""
    scores, magnitudes = [], []
    for sample in samples:
        p_med = sample[key]["prob"]["mediated"]
        p_unmed = sample[key]["prob"]["unmediated"]
        scores.append(p_med > p_unmed)
        magnitudes.append(p_med - p_unmed)
    score = sum(scores) / len(scores)
    magnitude = sum(magnitudes) / len(magnitudes)
    return EfficacyMetrics(score=score, magnitude=magnitude)


def paraphrase(sample: Sequence[dict]) -> EfficacyMetrics:
    """Computer paraphrase efficacy."""
    return efficacy(sample, key="paraphrase_prompts")


def consistency(
    generations: Sequence[StrSequence],
    references: Sequence[StrSequence],
    vectorizer: TfidfVectorizer,
) -> Metric:
    """Compute consistency score."""
    _validate_same_length(generations=generations, references=references)
    similarities = [
        _tfidf_similarity(gs, rs, vectorizer) for gs, rs in zip(generations, references)
    ]
    return Metric.aggregate(similarities)


def fluency(generations: Sequence[StrSequence], **kwargs: Any) -> Metric:
    """Compute fluency score.

    Args:
        generations: Any number of generated strings from the model.

    Returns:
        Average n-gram entropy across generations. Can specify multiple ns by
        setting the ns= kwarg. By default, uses 2-grams and 3-grams.

    """
    entropies = [_aggregate_n_gram_entropy(texts, **kwargs) for texts in generations]
    return Metric.aggregate(entropies)
