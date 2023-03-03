"""Compute before/after probabilities of concept attributes under REMEDI."""
import argparse
import json
import logging
from pathlib import Path

from remedi import benchmarks, data, editors, models, precompute
from remedi.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def _prefix_context(sample: dict) -> dict:
    """Prepend context to all prompts used in the eval."""
    entity = sample["entity"]
    context = sample["context"]

    source = {**sample["source"]}
    for key in ("all_co_features", "original_features"):
        source[key] = [
            {
                **feature,
                "prompt": precompute.prompt_in_context_from_sample(
                    entity, feature["prompt"], context
                ),
            }
            for feature in source[key]
        ]

    return {"source": source}


def main(args: argparse.Namespace) -> None:
    """Run the eval."""
    experiment = experiment_utils.setup_experiment(args)
    data.disable_caching()
    logging_utils.configure(args=args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    layers = args.layers
    if layers is None:
        layers = models.determine_layers(mt)

    if args.small:
        split = "train[5000:6000]"
    else:
        split = "train[5000:10000]"
    dataset = data.load_dataset("mcrae", split=split)

    baseline = args.baseline
    if baseline is not None:
        for banned in ("layers", "editors_dir"):
            if getattr(args, banned, None) is not None:
                raise ValueError(f"cannot set --{banned} with --baseline")

        if baseline == "prefix":
            dataset = dataset.map(_prefix_context, desc="prefix context")
        else:
            raise ValueError(f"unknown baseline: {baseline}")

        # Not used, but set so everything still runs.
        layers = [0]

        logger.info(f"will run {baseline} baseline")

    for layer in layers:
        benchmark_kwargs: dict = dict(dataset=dataset, device=device)
        if baseline is not None:
            benchmark_kwargs["mt"] = mt
            results_file = experiment.results_dir / baseline / "entailment.json"
        else:
            editor = editors.load_editor(
                mt, args.editor_type, layer, editors_dir=args.editors_dir, device=device
            )
            if editor is None:
                logger.warning(f"skipping benchmark for layer {layer}")
                continue

            benchmark_kwargs["editor"] = editor
            results_file = (
                experiment.results_dir
                / args.editor_type
                / str(layer)
                / "entailment.json"
            )

            logger.info(f"eval {args.editor_type}, layer {layer}")

        if results_file.exists():
            logger.info(f"found existing results, skipping")
            continue

        results = benchmarks.mcrae_entailment(**benchmark_kwargs)

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
    parser.add_argument(
        "--baseline",
        choices=("prefix", "replace"),
        help="run a baseline instead of evaluating an editor",
    )
    parser.add_argument("--small", action="store_true", help="run on subset of data")
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
