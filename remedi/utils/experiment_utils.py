"""Utilities for managing experiment runtimes and results."""
import argparse
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from remedi.utils import env_utils
from remedi.utils.typing import PathLike

import numpy
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEED = 123456


@dataclass(frozen=True)
class Experiment:
    """A configured experiment."""

    name: str
    results_dir: Path
    seed: int


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
    args_file_name: str | None = None,
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
        if args_file_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args_file_name = f"args-{timestamp}.json"
        args_file = results_dir / args_file_name
        logger.info(f"saving args to {args_file}")
        with args_file.open("w") as handle:
            json.dump({key: str(value) for key, value in vars(args).items()}, handle)

    return results_dir


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Add args common to all experiments.

    The args include:
        --experiment-name (-n): Requied, unique identifier for this experiment.
        --results-dir: Root directory containing all experiment folders.
        --clear-results-dir: If set, experiment-specific results directory is cleared.
        --args-file-name: Dump all args to this file; defaults to generated name.
        --seed: Random seed.

    """
    parser.add_argument(
        "--experiment-name",
        "-n",
        required=True,
        help="unique name for the experiment",
    )
    parser.add_argument(
        "--results-dir", type=Path, help="root directory containing experiment results"
    )
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        default=False,
        help="clear any old results and start anew",
    )
    parser.add_argument("--args-file-name", help="file name for args dump")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")


def setup_experiment(args: argparse.Namespace) -> Experiment:
    """Configure experiment from the args."""
    experiment_name = args.experiment_name
    seed = args.seed

    logger.info(f"setting up experiment {experiment_name}")

    set_seed(seed)

    results_dir = create_results_dir(
        experiment_name,
        root=args.results_dir,
        args=args,
        args_file_name=args.args_file_name,
        clear_if_exists=args.clear_results_dir,
    )

    return Experiment(
        name=experiment_name,
        results_dir=results_dir,
        seed=seed,
    )
