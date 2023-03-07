"""Evaluate direction classification in Bias in Bios setting."""
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
    dataset = data.load_dataset("biosbias", split=split)
    dataset = precompute.from_args(args, dataset)

    # Precompute model's predictions on prompt in context, no editing.
    labels = sorted({x["target_mediated"] for x in dataset})
    dataset = precompute.prompt_in_context_from_dataset(dataset)
    dataset = precompute.model_predictions_from_dataset(
        mt,
        dataset,
        other_targets=labels,
        input_prompt_key="prompt_in_context",
        input_target_key=None,
        input_comparator_key=None,
        device=device,
        batch_size=args.batch_size,
        desc=f"error classification [model predictions]",
    )

    editor_layers = args.layers
    editor_type = args.editor_type
    editors_dir = args.editors_dir

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
            results_file_name = f"error_cls_layer_{entity_layer}"
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
            results = benchmarks.biosbias_error_classification(
                editor=editor,
                dataset=dataset,
                device=device,
                labels=labels,
                control_task=args.control_task,
                entity_layer=entity_layer,
            )
            logging.info(
                f"benchmark complete! results:\n%s",
                json.dumps(results.metrics.to_dict(), indent=1),
            )

            results_file.parent.mkdir(exist_ok=True, parents=True)
            with results_file.open("w") as handle:
                json.dump(results.to_dict(), handle)

            metrics_file = results_file.parent / f"{results_file.stem}_metrics.json"
            with metrics_file.open("w") as handle:
                json.dump(results.metrics.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate classification in bias setting"
    )
    parser.add_argument(
        "--entity-layers", nargs="+", type=int, help="entity layers to probe at"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=editors.DEFAULT_BATCH_SIZE,
        help="model batch size",
    )
    parser.add_argument(
        "--control-task", default=False, action="store_true", help="use control task"
    )
    parser.add_argument(
        "--control-model",
        default=False,
        action="store_true",
        help="assume input model is control model",
    )
    parser.add_argument(
        "--small", default=False, action="store_true", help="use subset of data"
    )
    # No data args because this only works on biosbias.
    models.add_model_args(parser)
    editors.add_editor_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    precompute.add_preprocessing_args(parser)
    args = parser.parse_args()
    main(args)
