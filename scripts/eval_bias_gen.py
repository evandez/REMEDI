"""Evaluate editor effects on generation for bias setting."""
import argparse
import json
import logging
from pathlib import Path

from src import benchmarks, data, editors, models, precompute
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

    dataset = data.load_dataset("biosbias", split="train[5000:5250]")

    editors_dir = args.editors_dir
    editor_type = args.editor_type
    layers = args.layers
    if layers is None:
        layers = editors.list_saved_editors(editors_dir)[editor_type]
    logger.info(f"found editors for layers: {layers}")

    benchmark_kwargs: dict = {}
    if args.decontextualized:
        dataset = precompute.prompt_in_context_from_dataset(dataset)
        benchmark_kwargs["prompt_key"] = "prompt"

    baseline_results_file = experiment.results_dir / "baseline.json"
    if not baseline_results_file.exists():
        logger.info("begin baseline")
        baseline_results = benchmarks.error_correction(
            mt=mt,
            dataset=dataset,
            device=device,
            desc="error correction [baseline]",
            **benchmark_kwargs,
        )
        logging.info(
            f"baseline complete! results:\n%s",
            json.dumps(baseline_results.metrics.to_dict(), indent=1),
        )
        baseline_results_file.parent.mkdir(exist_ok=True, parents=True)
        with baseline_results_file.open("r") as handle:
            json.dump(baseline_results.to_dict(), handle)
    else:
        logger.info(
            f"existing baseline results found at {baseline_results_file}; skipping"
        )

    for layer in layers:
        results_file = (
            experiment.results_dir / editor_type / str(layer) / "error_correction.json"
        )
        if results_file.exists() and not args.rerun:
            logger.info(f"found existing results for layer {layer} at {results_file}")
            continue

        editor = editors.load_editor(
            mt, editor_type, layer, editors_dir=editors_dir, device=device
        )
        if editor is None:
            logger.warning(f"skipping benchmark for layer {layer}")
            continue

        logger.info(f"begin eval for layer {layer}")
        results = benchmarks.error_correction(
            editor=editor,
            dataset=dataset,
            device=device,
            **benchmark_kwargs,
        )

        logging.info(
            f"benchmark complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )

        results_file.parent.mkdir(exist_ok=True, parents=True)
        with results_file.open("w") as handle:
            json.dump(results.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate editor generation on bias dataset"
    )
    parser.add_argument("--editor-type", "-t", default="linear", help="editor type")
    parser.add_argument(
        "--editors-dir",
        "-e",
        type=Path,
        help="path to editor experiment",
    )
    parser.add_argument(
        "--layers", "-l", nargs="+", type=int, help="layers to test editors for"
    )
    parser.add_argument(
        "--decontextualized",
        action="store_true",
        help="evaluate in decontextualized setting",
    )
    parser.add_argument("--rerun", action="store_true", help="force rerun all evals")
    # No dataset args because this only works on biosbias
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
