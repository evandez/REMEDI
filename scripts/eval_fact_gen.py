"""Evaluate editors on the Counterfact benchmark."""
import argparse
import json
import logging
from pathlib import Path

from src import benchmarks, data, editors, models
from src.utils import experiment_utils, logging_utils
from src.utils.typing import Device

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def load_editor(
    editor_type: str,
    mt: models.ModelAndTokenizer,
    layer: int,
    editors_dir: Path | None = None,
    device: Device | None = None,
) -> editors.Editor | None:
    """Load editor of given type from the directory, assuming default options."""
    editor_factory = editors.SUPPORTED_EDITORS[editor_type]
    editor = editor_factory(mt=mt, layer=layer)
    editor.to(device)

    if editor_type != "identity":
        if editors_dir is None:
            logger.warning("editors_dir not specified for non-identity editor")
            return None

        weights_file = editors_dir / editor_type / str(layer) / "weights.pth"
        if not weights_file.exists():
            logger.warning(f"weights expected at {weights_file} but not found")
            return None

        logger.info(f"loading editor weights from {weights_file}")
        state_dict = torch.load(weights_file, map_location=device)
        editor.load_state_dict(state_dict)

    return editor


def main(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16

    editors_dir = args.editors_dir
    editor_type = args.editor_type
    if editor_type != "identity":
        logger.info(f"will look for {editor_type} editors in {editors_dir}")
        if not Path(editors_dir, editor_type).exists():
            raise ValueError(f"editors not found at {editors_dir}")

    layers = args.layers
    if layers is None:
        layers = sorted(
            [
                int(layer_dir.name)
                for layer_dir in editors_dir.iterdir()
                if layer_dir.is_dir()
            ]
        )

    logger.info(f"loading {args.model} (device={device}, fp16={fp16})")
    mt = models.load_model(args.model, device=device, fp16=fp16)

    logger.info("loading several data sources")
    dataset = data.load_dataset("counterfact", split="train[5000:]")
    attribute_snippets = data.load_attribute_snippets()
    tfidf_vectorizer = data.load_tfidf_vectorizer()

    for layer in layers:
        editor = load_editor(
            editor_type, mt, layer, editors_dir=editors_dir, device=device
        )
        if editor is None:
            logger.warning(f"skipping benchmark for layer {layer}")
            continue

        results: (
            benchmarks.EfficacyBenchmarkResults
            | benchmarks.ParaphraseBenchmarkResults
            | benchmarks.GenerationBenchmarkResults
            | benchmarks.EssenceBenchmarkResults
        )
        for benchmark_name in args.benchmarks:
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
            results_file = (
                experiment.results_dir / str(layer) / f"{benchmark_name}.json"
            )
            with results_file.open("w") as handle:
                json.dump(results, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate editors")
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        choices=(
            "efficacy",
            "paraphrase",
            "generation",
            "essence",
        ),
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
        "--model",
        "-m",
        choices=models.SUPPORTED_MODELS,
        default=models.GPT_J_NAME,
        help="model to classify on",
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
    parser.add_argument("--fp16", action="store_true", help="use fp16 model version")
    parser.add_argument("--device", help="device to run model on")
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    data.add_dataset_args(parser)
    args = parser.parse_args()
    main(args)
