"""Compute before/after probabilities of concept attributes under REMEDI."""
import argparse
import json
import logging
from pathlib import Path

from remedi import benchmarks, data, editors, models
from remedi.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the eval."""
    experiment = experiment_utils.setup_experiment(args)
    data.disable_caching()
    logging_utils.configure(args=args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    if args.small:
        split = "train[5000:5100]"
    else:
        split = "train[5000:6000]"
    dataset = data.load_dataset("mcrae", split=split)

    layers = args.layers
    if layers is None:
        layers = models.determine_layers(mt)

    for layer in layers:
        results_file = (
            experiment.results_dir / args.editor_type / str(layer) / f"entailment.json"
        )
        if results_file.exists():
            logger.info(f"found existing results for layer {layer}, skipping")
            continue

        logger.info(f"begin eval for layer {layer}")
        editor = editors.load_editor(
            mt, args.editor_type, layer, editors_dir=args.editors_dir, device=device
        )
        if editor is None:
            logger.warning(f"skipping benchmark for layer {layer}")
            continue

        results = benchmarks.mcrae_entailment(
            editor=editor, dataset=dataset, device=device
        )

        logger.info(
            f"results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )

        results_file.parent.mkdir(exist_ok=True, parents=True)
        with results_file.open("w") as handle:
            json.dump(results.to_dict(), handle)

        metrics_file = results_file.parent / f"{results_file.stem}_metrics.json"
        with metrics_file.open("w") as handle:
            json.dump(results.metrics.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run mcrae eval")
    parser.add_argument(
        "--editor-type", "-t", default="linear", help="editor type, inferred by default"
    )
    parser.add_argument(
        "--editors-dir", "-e", type=Path, help="path to editor experiment"
    )
    parser.add_argument(
        "--layers", "-l", nargs="+", type=int, help="layers to apply remedi at"
    )
    parser.add_argument("--small", action="store_true", help="run on subset of data")
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
