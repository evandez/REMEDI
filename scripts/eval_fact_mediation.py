"""Evaluate how often LM mediates factual information in and out of context."""
import argparse
import json
import logging

from remedi import benchmarks, data, models
from remedi.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the mediation evaluation."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    dataset = data.load_dataset("counterfact", split="train")

    results_file = experiment.results_dir / "mediation.json"
    if results_file.exists():
        logger.info(f"found existing results at {results_file}")
        exit()

    results = benchmarks.mediation(mt=mt, dataset=dataset, device=device)

    decontextual = results.decontextual
    assert decontextual is not None
    logger.info(
        "decontextual results:\n%s",
        json.dumps(decontextual.metrics.to_dict(), indent=1),
    )

    contextual = results.contextual
    assert contextual is not None
    logger.info(
        "contextual results:\n%s",
        json.dumps(contextual.metrics.to_dict(), indent=1),
    )

    results_file.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"writing results to {results_file}")
    with results_file.open("w") as handle:
        json.dump(results.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate mediation")
    # No dataset args because this only works with counterfact for now.
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
