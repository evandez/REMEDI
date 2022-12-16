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
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metric(DataClassJsonMixin):
    """An aggregate metric."""

    mean: float
    std: float

    @staticmethod
    def aggregate(values: ArrayLike) -> "Metric":
        return Metric(np.mean(values), np.std(values))


@dataclass(frozen=True)
class EfficacyMetrics(DataClassJsonMixin):
    """Efficacy metrics."""

    score: Metric
    magnitude: Metric


def efficacy(
    p_targets: Sequence[ArrayLike],
    p_comparators: Sequence[ArrayLike],
    assume_log_probs: bool = True,
) -> EfficacyMetrics:
    """Compute efficacy on metrics.

    Efficacy is determined by a score and a magnitude. The score is how often
    p(target) > p(comparator). The magnitude is the average p(target) - p(comparator).

    Inputs are two sequences. Each element should be one or more measurements of
    the probability (for e.g. different prompts). This function will first average
    across those inner lists, then average across the whole list.
    """
    _validate_same_length(p_targets=p_targets, p_comparators=p_comparators)

    scores, magnitudes = [], []
    for i, (p_target, p_comparator) in enumerate(zip(p_targets, p_comparators)):
        _validate_same_length(
            **{f"p_target_{i}": p_target, f"p_comparator_{i}": p_comparator}
        )

        if assume_log_probs:
            p_target = np.exp(p_target)
            p_comparator = np.exp(p_comparator)
        else:
            p_target = np.array(p_target)
            p_comparator = np.array(p_comparator)

        score = np.mean(p_target > p_comparator)
        scores.append(score)

        magnitude = np.mean(p_target - p_comparator)
        magnitudes.append(magnitude)

    return EfficacyMetrics(
        score=Metric.aggregate(scores), magnitude=Metric.aggregate(magnitudes)
    )


# TODO(evandez): Move all the TF-IDF stuff to this file.


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


def _validate_same_length(**kwargs: Sequence | ArrayLike) -> None:
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
