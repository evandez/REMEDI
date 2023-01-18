"""Evaluate editors on the Counterfact benchmark."""
import argparse
import json
import logging
from pathlib import Path

from src import benchmarks, data, editors, models, precompute
from src.utils import experiment_utils, logging_utils

import torch
import torch.utils.data

logger = logging.getLogger(__name__)

BENCHMARKS = (
    "efficacy",
    "paraphrase",
    "generation",
    "essence",
)


def main(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading {args.model} (device={device}, fp16={args.fp16})")
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    logger.info("loading several data sources")
    dataset = data.load_dataset("counterfact", split="train[5000:5250]")
    attribute_snippets = data.load_attribute_snippets()
    tfidf_vectorizer = data.load_tfidf_vectorizer()

    baseline = args.baseline
    if baseline is not None:
        for banned in ("layers", "editors_dir"):
            if getattr(args, banned, None) is not None:
                raise ValueError(f"cannot set --{banned} with --baseline")

        if baseline == "prepend-context":
            dataset = precompute.prompt_in_context_from_dataset(
                dataset, output_key="prompt"
            )
        else:
            raise ValueError(f"unknown baseline: {baseline}")

        # Not used, but set so everything still runs.
        editors_dir = None
        editor_type = "null"
        layers = [0]

        logger.info(f"will run {baseline} baseline")
    else:
        editor_type = args.editor_type
        editors_dir = args.editors_dir
        if editor_type not in {"null", "identity"} and editors_dir is None:
            raise ValueError("--editors-dir required when evaluating non-null editor")

        layers = args.layers
        if layers is None:
            layers = editors.list_saved_editors(args.editors_dir)[editor_type]
        logger.info(f"found {editor_type} editors for layers: {layers}")

    for layer in layers:
        benchmark_kwargs: dict = dict(dataset=dataset, device=device)
        if baseline is None:
            editor = editors.load_editor(
                mt, editor_type, layer, editors_dir=editors_dir, device=device
            )
            if editor is None:
                logger.warning(f"skipping benchmark for layer {layer}")
                continue
            benchmark_kwargs["editor"] = editor
            logger.info(f"eval {editor_type} editor, layer {layer}")
        else:
            benchmark_kwargs["mt"] = mt
            logger.info(f"eval {baseline} baseline")

        results: (
            benchmarks.EfficacyBenchmarkResults
            | benchmarks.CounterFactParaphraseBenchmarkResults
            | benchmarks.CounterFactGenerationBenchmarkResults
            | benchmarks.EssenceBenchmarkResults
        )
        for benchmark_name in args.benchmarks:
            if baseline is not None:
                results_file = experiment.results_dir / f"{baseline}-baseline.json"
            else:
                results_file = (
                    experiment.results_dir
                    / editor_type
                    / str(layer)
                    / f"{benchmark_name}.json"
                )
            if results_file.exists() and not args.rerun:
                logger.info(
                    f"found existing {benchmark_name} results for layer {layer} "
                    f"at {results_file}"
                )
                continue

            if benchmark_name == "efficacy":
                results = benchmarks.efficacy(**benchmark_kwargs)
            elif benchmark_name == "paraphrase":
                results = benchmarks.counterfact_paraphrase(**benchmark_kwargs)
            elif benchmark_name == "generation":
                results = benchmarks.counterfact_generation(
                    attribute_snippets=attribute_snippets,
                    tfidf_vectorizer=tfidf_vectorizer,
                    **benchmark_kwargs,
                )
            elif benchmark_name == "essence":
                results = benchmarks.essence(
                    tfidf_vectorizer=tfidf_vectorizer, **benchmark_kwargs
                )
            else:
                raise ValueError(f"unknown benchmark: {benchmark_name}")

            logging.info(
                f"{benchmark_name} benchmark complete! results:\n%s",
                json.dumps(results.metrics.to_dict(), indent=1),
            )
            results_file.parent.mkdir(exist_ok=True, parents=True)
            with results_file.open("w") as handle:
                json.dump(results.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate editors")
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        help="benchmarks to run, defaults depend on dataset",
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
        "--baseline",
        choices=("prepend-context",),
        help="run a baseline instead of evaluating an editor",
    )
    parser.add_argument("--rerun", action="store_true", help="force rerun all evals")
    # No dataset args because this only works for counterfact
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
