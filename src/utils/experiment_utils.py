"""Utilities for managing experiments and results."""
import argparse
import json
import logging
import random
import shutil
from pathlib import Path

from src.utils import env_utils
from src.utils.typing import PathLike

import numpy
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    logger.info("setting all seeds to %d", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def create_results_dir(
    experiment_name: str,
    root: PathLike | None = None,
    args: argparse.Namespace | None = None,
    args_file: PathLike | None = None,
    clear_if_exists: bool = False,
) -> Path:
    """Create a directory for storing experiment results.

    Args:
        name: Experiment name.
        root: Root directory to store results in. Consults env if not set.
        args: If set, save the full argparse namespace as JSON.
        args_file: Save args file here.
        clear_if_exists: Clear the results dir if it already exists.

    Returns:
        The initialized results directory.

    """
    if root is None:
        root = env_utils.determine_results_dir()
    root = Path(root)

    results_dir = root / experiment_name
    results_dir = results_dir.resolve()

    if results_dir.exists():
        logger.info(f"rerunning experiment {experiment_name}")
        if clear_if_exists:
            logger.info(f"clearing previous results from {results_dir}")
            shutil.rmtree(results_dir)

    results_dir.mkdir(exist_ok=True, parents=True)
    if args is not None:
        if args_file is None:
            args_file = results_dir / "args.json"
        args_file = Path(args_file)
        logger.info(f"saving results to {args_file}")
        with args_file.open("w") as handle:
            json.dump(vars(args), handle)

    return results_dir
