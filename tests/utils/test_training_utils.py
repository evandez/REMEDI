"""Unit tests for the `src.utils.training_utils` module."""
from remedi.utils import training_utils

PATIENCE = 5


def test_early_stopping_init_decreasing():
    """Test EarlyStopping.__init__ records when value should decrease."""
    early_stopping = training_utils.EarlyStopping(patience=PATIENCE, decreasing=True)
    assert early_stopping.patience == PATIENCE
    assert early_stopping.decreasing is True
    assert early_stopping.best == float("inf")
    assert early_stopping.num_bad == 0


def test_early_stopping_init_increasing():
    """Test EarlyStopping.__init__ records when value should increase."""
    early_stopping = training_utils.EarlyStopping(
        patience=PATIENCE,
        decreasing=False,
    )
    assert early_stopping.patience == PATIENCE
    assert early_stopping.decreasing is False
    assert early_stopping.best == float("-inf")
    assert early_stopping.num_bad == 0


def test_early_stopping_call_decreasing():
    """Test EarlyStopping.__call__ returns when value does not decrease."""
    early_stopping = training_utils.EarlyStopping(patience=PATIENCE, decreasing=True)
    assert not early_stopping(-1)
    for i in range(PATIENCE):
        assert not early_stopping(i)
    assert early_stopping(0)


def test_early_stopping_call_increasing():
    """Test EarlyStopping.__call__ reports when value does not increases."""
    early_stopping = training_utils.EarlyStopping(
        patience=PATIENCE,
        decreasing=False,
    )
    assert not early_stopping(PATIENCE + 1)
    for i in range(PATIENCE):
        assert not early_stopping(i)
    assert early_stopping(0)


def test_early_stopping_improved():
    """Test EarlyStopping.improved returns True when value improves."""
    early_stopping = training_utils.EarlyStopping(patience=PATIENCE, decreasing=True)

    early_stopping(0)
    assert early_stopping.improved

    early_stopping(1)
    assert not early_stopping.improved

    early_stopping(-1)
    assert early_stopping.improved
