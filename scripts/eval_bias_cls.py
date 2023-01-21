"""Evaluate direction classification in Bias in Bios setting."""
import argparse
import logging

from src import data, editors, models, precompute
from src.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the evaluation."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading {args.model} (device={device}, fp16={args.fp16})")
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    if args.small:
        split = "train[5000:6000]"
    else:
        split = "train[5000:10000]"
    dataset = data.load_dataset("biosbias", split=split)
    dataset = precompute.from_args(args, dataset)

    # TODO(evandez): Finish.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate classification in bias setting"
    )
    # No data args because this only works on biosbias.
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    precompute.add_preprocessing_args(parser)
    args = parser.parse_args()
    main(args)
