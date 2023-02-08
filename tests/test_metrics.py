"""Simple unit tests for metrics."""
from remedi import metrics

import numpy
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.mark.parametrize(
    "values,mean,std",
    (
        ([1, 2, 3], 2, 0.8164965),
        ([1], 1, 0),
    ),
)
def test_metric_aggregate(values, mean, std):
    """Test `Metric.aggregate` corrctly computes mean and std."""
    actual = metrics.Metric.aggregate(values)
    assert numpy.allclose(actual.mean, mean, atol=1e-5)
    assert numpy.allclose(actual.std, std, atol=1e-5)


@pytest.mark.parametrize(
    "p_targets,p_comparators,score,magnitude",
    (
        ([[1]], [[0.5]], 1.0, 0.5),
        ([[0, 1], [0.5, 0.75, 1]], [[0.5, 0.5], [0.5, 0.75, 1]], 0.25, 0),
    ),
)
def test_efficacy(p_targets, p_comparators, score, magnitude):
    """Test efficacy correctly computes efficacy."""
    actual = metrics.efficacy(p_targets, p_comparators, assume_log_probs=False)
    assert numpy.allclose(actual.score.mean, score)
    assert numpy.allclose(actual.magnitude.mean, magnitude)


@pytest.fixture
def tfidf_vectorizer():
    """Create a TF-IDF vectorizer for testing."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["dog cat"])
    return vectorizer


def test_consistency(tfidf_vectorizer):
    """Test average_tfidf_similarity does something...sensible."""
    actual = metrics.average_tfidf_similarity(
        [["dog cat"]], [["dog dog dog", "cat cat cat"]], tfidf_vectorizer
    )
    assert numpy.allclose(actual.mean, 1.0, atol=1e-4)


@pytest.mark.parametrize(
    "texts,fluency",
    (
        ([["a a a"]], 0),
        ([["a"]], 0),
        ([["a b " * 100]], 1),
        ([["a", "b"], ["a b " * 100]], 0.5),
    ),
)
def test_average_weighted_n_gram_entropy(texts, fluency):
    """Test average_weighted_n_gram_entropy correctly computes entropy."""
    actual = metrics.average_weighted_n_gram_entropy(texts)
    assert numpy.allclose(actual.mean, fluency, atol=1e-4)
