"""Evaluate editors on the CounterFact benchmark."""
import argparse
import logging
from collections import cast, defaultdict
from pathlib import Path

from src import data, editors, metrics, models
from src.utils import env_utils, experiment_utils

import torch
import torch.utils.data
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    experiment_utils.set_seed(args.seed)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16

    experiment_name = args.experiment_name or "benchmark"
    results_dir = experiment_utils.create_results_dir(
        experiment_name,
        root=args.results_dir,
        args=args,
        clear_if_exists=args.clear_results_dir,
    )

    editors_dir = args.editors
    if editors_dir is None:
        editors_dir = env_utils.determine_results_dir() / "editors"
    logger.info(f"will look for editors in {editors_dir}")
    if not editors_dir.exists():
        raise ValueError(f"editors not found at {editors_dir}; maybe pass the -e flag")

    logger.info(f"loading {args.model} (device={device}, fp16={fp16})")
    mt = models.load_model(args.model, device=device, fp16=fp16)

    logger.info("loading several data sources")
    # TODO(evandez): Use full counterfact after splitting properly.
    dataset = data.load_dataset("counterfact", split="train[5000:10000]")
    snippets = data.load_attribute_snippets()
    vectorizer = data.load_tfidf_vectorizer()

    probs = defaultdict(list)
    generations = defaultdict(list)
    with dataset.formatted_as("torch"):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset), batch_size=args.batch_size
        )
        for batch in tqdm(loader, desc="generate text"):
            # TODO(evandez): Implement.
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run full counterfact benchmark")
    parser.add_argument("--name", "-n", help="experiment name")
    parser.add_argument("--editors", "-e", type=Path, help="path to editor experiment")
    parser.add_argument(
        "--layers", "-l", nargs="+", type=int, help="layers to test editors for"
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=models.SUPPORTED_MODELS,
        default=models.GPT_J_NAME,
        help="model to classify on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=editors.DEFAULT_BATCH_SIZE,
        help="model batch size",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16 model version")
    parser.add_argument("--device", help="device to run model on")
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        help="clear old results and start anew",
    )
    args = parser.parse_args()
    main(args)
