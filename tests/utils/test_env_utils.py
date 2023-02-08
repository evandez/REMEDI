"""Unit tests for the src.utils.env module."""
import os
import pathlib

from remedi.utils import env_utils

import pytest


@pytest.fixture
def root():
    """Return path to root of repo."""
    return pathlib.Path(__file__).parents[2]


@pytest.mark.parametrize(
    "path,expected,strict",
    (
        ("foo/bar", "foo/bar", False),
        ("/tmp/foo/bar", "/tmp/foo/bar", True),
    ),
)
def test_maybe_relative_to_repo(root, path, expected, strict):
    """Test maybe_relative_to_repo resolves path correctly."""
    actual = env_utils.maybe_relative_to_repo(path)
    if strict:
        assert str(actual) == expected
    else:
        assert str(actual).endswith(f"{root.name}/{expected}")


@pytest.mark.parametrize(
    "name,default,environ,expected,strict",
    (
        (
            "TEST_ENV_EXISTS",
            "unused",
            {"TEST_ENV_EXISTS": "/foo/bar", "TEST_ENV_ALSO_EXISTS": "baz"},
            "/foo/bar",
            True,
        ),
        (
            "NEEDS_DEFAULT",
            "foo/bar",
            {"TEST_ENV_EXISTS": "foo/bar", "TEST_ENV_ALSO_EXISTS": "baz"},
            "foo/bar",
            False,
        ),
    ),
)
def test_read_path(root, name, default, environ, expected, strict):
    """Test read_path correctly obtains path from environment."""
    os.environ.update(environ)
    actual = env_utils.read_path(name, default)
    if strict:
        assert str(actual) == expected
    else:
        assert str(actual).endswith(f"{root.name}/{expected}")

    # Reset, just in case...
    for key in environ:
        del os.environ[key]


@pytest.mark.parametrize(
    "fn,var,expected",
    (
        (
            env_utils.determine_data_dir,
            env_utils.ENV_DATA_DIR,
            env_utils.DEFAULT_DATA_DIR,
        ),
        (
            env_utils.determine_results_dir,
            env_utils.ENV_RESULTS_DIR,
            env_utils.DEFAULT_RESULTS_DIR,
        ),
    ),
)
def test_default_dirs(root, fn, var, expected):
    """Test data_dir/models_dir/results_dir returns correct default."""
    if var in os.environ:
        del os.environ[var]
    actual = fn()
    assert str(actual).endswith(f"{root.name}/{expected}")
