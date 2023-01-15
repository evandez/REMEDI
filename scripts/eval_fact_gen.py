"""Evaluate editors on the Counterfact benchmark."""
import argparse
import json
import logging
from pathlib import Path

from src import benchmarks, data, editors, models
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
    fp16 = args.fp16
    editors_dir = args.editors_dir
    editor_type = args.editor_type

    layers = args.layers
    if layers is None:
        layers = editors.list_saved_editors(editors_dir)[editor_type]
    logger.info(f"found editors for layers: {layers}")

    logger.info(f"loading {args.model} (device={device}, fp16={fp16})")
    mt = models.load_model(args.model, device=device, fp16=fp16)

    logger.info("loading several data sources")
    dataset = data.load_dataset("counterfact", split="train[5000:]")
    attribute_snippets = data.load_attribute_snippets()
    tfidf_vectorizer = data.load_tfidf_vectorizer()

    for layer in layers:
        editor = editors.load_editor(
            mt, editor_type, layer, editors_dir=editors_dir, device=device
        )
        if editor is None:
            logger.warning(f"skipping benchmark for layer {layer}")
            continue
        logger.info(f"begin eval for layer {layer}")

        results: (
            benchmarks.EfficacyBenchmarkResults
            | benchmarks.CounterFactParaphraseBenchmarkResults
            | benchmarks.CounterFactGenerationBenchmarkResults
            | benchmarks.EssenceBenchmarkResults
        )
        for benchmark_name in args.benchmarks:
            results_file = (
                experiment.results_dir / str(layer) / f"{benchmark_name}.json"
            )
            if results_file.exists() and not args.rerun:
                logger.info(
                    f"found existing {benchmark_name} results for layer {layer} "
                    f"at {results_file}"
                )
                continue

            if benchmark_name == "efficacy":
                results = benchmarks.efficacy(
                    editor=editor, dataset=dataset, device=device
                )
            elif benchmark_name == "paraphrase":
                results = benchmarks.counterfact_paraphrase(
                    editor=editor, dataset=dataset, device=device
                )
            elif benchmark_name == "generation":
                results = benchmarks.counterfact_generation(
                    editor=editor,
                    dataset=dataset,
                    attribute_snippets=attribute_snippets,
                    tfidf_vectorizer=tfidf_vectorizer,
                    max_length=args.max_length,
                )
            elif benchmark_name == "essence":
                results = benchmarks.essence(
                    editor=editor,
                    dataset=dataset,
                    device=device,
                    tfidf_vectorizer=tfidf_vectorizer,
                )
            else:
                raise ValueError(f"unknown benchmark: {benchmark_name}")

            logging.info(
                f"{benchmark_name} benchmark complete! results:\n%s",
                json.dumps(results.metrics.to_dict(), indent=1),
            )
            results_file.parent.mkdir(exist_ok=True, parents=True)
            with results_file.open("w") as handle:
                json.dump(results, handle)


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
    parser.add_argument("--editor-type", "-t", help="editor type, inferred by default")
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
        "--prepend-context", "-p", action="store_true", help="prepend context to prompt"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=editors.DEFAULT_BATCH_SIZE,
        help="model batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=editors.DEFAULT_MAX_LENGTH,
        help="number of tokens to generate including prompt",
    )
    parser.add_argument("--rerun", action="store_true", help="force rerun all evals")
    # No dataset args because this only works for counterfact
    models.add_model_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
