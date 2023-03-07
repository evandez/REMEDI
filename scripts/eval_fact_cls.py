"""Evaluate editor's ability to classify fact mediation."""
import argparse
import json
import logging
from pathlib import Path

from remedi import benchmarks, data, editors, models, precompute
from remedi.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the evaluation."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)
    n_layers = len(models.determine_layers(mt))

    if args.small:
        split = "train[5000:6000]"
    else:
        split = "train[5000:10000]"
    dataset = data.load_dataset("counterfact", split=split)
    dataset = precompute.from_args(args, dataset)

    editors_dir = args.editors_dir
    editor_type = args.editor_type
    editor_layers = args.layers
    if editor_layers is None:
        editor_layers = editors.list_saved_editors(editors_dir)[editor_type]
    logger.info(f"found editors for layers: {editor_layers}")

    for editor_layer in editor_layers:
        editor = editors.load_editor(
            mt, editor_type, editor_layer, editors_dir=editors_dir, device=device
        )
        if editor is None:
            logger.warning(f"skipping benchmark for editor layer {editor_layer}")
            continue

        entity_layers = args.entity_layers
        if entity_layers is None:
            entity_layers = range(editor_layer, n_layers)
        logger.info(f"will probe at layers: {entity_layers}")

        for entity_layer in entity_layers:
            results_file_name = f"fact_cls_layer_{entity_layer}"
            if args.control_task:
                results_file_name = f"{results_file_name}_control_task"
            if args.control_model:
                results_file_name = f"{results_file_name}_control_model"
            results_file = (
                experiment.results_dir
                / editor_type
                / str(editor_layer)
                / f"{results_file_name}.json"
            )
            if results_file.exists():
                logger.info(
                    f"found existing results for editor layer {editor_layer}, "
                    f"entity layer {entity_layer} at {results_file}"
                )
                continue

            logger.info(
                f"begin editor_layer={editor_layer}, entity_layer={entity_layer}"
            )
            results = benchmarks.classification(
                editor=editor,
                dataset=dataset,
                device=device,
                entity_layer=entity_layer,
                control_task=args.control_task,
            )

            for task_key in ("contextual", "decontextual"):
                metrics: benchmarks.ClassifierMetrics = getattr(
                    results.metrics, task_key
                )
                logger.info(
                    f"{task_key} results:\n%s",
                    json.dumps(metrics.to_dict(), indent=1),
                )

            results_file.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"writing results to {results_file}")
            with results_file.open("w") as handle:
                json.dump(results.to_dict(), handle)

            metrics_file = results_file.parent / f"{results_file.stem}_metrics.json"
            with metrics_file.open("w") as handle:
                json.dump(results.metrics.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate fact classification")
    parser.add_argument(
        "--entity-layers", nargs="+", type=int, help="layers to get probe entity at"
    )
    parser.add_argument(
        "--control-task",
        default=False,
        action="store_true",
        help="classify on control task",
    )
    parser.add_argument(
        "--control-model",
        default=False,
        action="store_true",
        help="assume input model is control model",
    )
    parser.add_argument(
        "--small", action="store_true", help="run on a small subset of data"
    )
    # No data args because this only works on CounterFact.
    models.add_model_args(parser)
    precompute.add_preprocessing_args(parser)
    editors.add_editor_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
