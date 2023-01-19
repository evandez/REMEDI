"""Evaluate editor's ability to classify fact mediation."""
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

    dataset = data.load_dataset("counterfact", split="train[5000:6000]")
    dataset = precompute.from_args(args, dataset)

    editors_dir = args.editors_dir
    editor_type = args.editor_type
    layers = args.layers
    if layers is None:
        layers = editors.list_saved_editors(editors_dir)[editor_type]
    logger.info(f"found editors for layers: {layers}")

    for layer in layers:
        results_file_name = "fact-cls"
        if args.control_task:
            results_file_name = f"{results_file_name}-control"
        results_file = (
            experiment.results_dir
            / editor_type
            / str(layer)
            / f"{results_file_name}.json"
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

        logger.info(f"begin classification for layer {layer}")
        results = benchmarks.classification(
            editor=editor,
            dataset=dataset,
            device=device,
            entity_layer=args.entity_layer,
            control_task=args.control_task,
        )

        for task_key in ("contextual", "decontextual"):
            metrics: benchmarks.ClassifierMetrics = getattr(results, task_key)
            logger.info(
                f"{task_key} results:\n%s",
                json.dumps(metrics.to_dict(), indent=1),
            )

        results_file.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"writing results to {results_file}")
        with results_file.open("w") as handle:
            json.dump(results.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate fact classification")
    parser.add_argument(
        "--editor-type", "-t", default="linear", help="editor type, inferred by default"
    )
    parser.add_argument(
        "--editors-dir",
        "-e",
        type=Path,
        help="path to editor experiment",
    )
    parser.add_argument(
        "--layers", "-l", nargs="+", type=int, help="layers to test editors for"
    )
    parser.add_argument("--entity-layer", type=int, help="layer to get entity rep from")
    parser.add_argument(
        "--control-task", action="store_true", help="classify on control task"
    )
    # No data args because this only works on CounterFact.
    models.add_model_args(parser)
    precompute.add_preprocessing_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
