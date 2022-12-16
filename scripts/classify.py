"""Classify attributes of entities using directions."""
import argparse
import logging
from pathlib import Path

from src import data, editors, models, precompute
from src.utils import env_utils, experiment_utils, logging_utils

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the classification."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16

    editors_dir = args.editors
    if editors_dir is None:
        editors_dir = env_utils.determine_results_dir() / "editors"
    if not editors_dir.exists():
        raise ValueError(f"editors not found at {editors_dir}; maybe pass the -e flag")
    editors_dir /= "linear"

    logger.info(f"loading {args.model} (device={device}, fp16={fp16})")
    mt = models.load_model(args.model, device=device, fp16=fp16)

    # TODO(evandez): Just make a dedicated train/test split for counterfact.
    train = data.load_dataset(args.dataset, split="train[:5000]")
    test = data.load_dataset(args.dataset, split="train[5000:10000]")

    layers = args.layers
    if layers is None:
        layers = sorted(
            [
                int(layer_dir.name)
                for layer_dir in editors_dir.iterdir()
                if layer_dir.is_dir()
            ]
        )

    for layer in layers:
        logger.info(f"begin layer {layer}")
        weights_file = editors_dir / str(layer) / "weights.pth"
        if not weights_file.exists():
            logger.info(f"no trained editor found; skipping")
            continue

        editor = editors.LinearEditor(mt=mt, layer=layer).to(device)
        state_dict = torch.load(weights_file, map_location=device)
        editor.load_state_dict(state_dict)

        for split, subset in (("train", train), ("test", test)):
            results_file = (
                experiment.results_dir / "linear" / str(layer) / f"{split}-class.json"
            )
            if results_file.exists():
                logger.info(f"found {split} results at {results_file}; skipping")
                continue

            precomputed = precompute.classification_inputs_from_dataset(
                mt=mt,
                dataset=subset,
                layers=[editor.layer],
                device=device,
                batch_size=args.batch_size,
                desc=f"precompute {split} directions",
            )
            results = editor.classify(
                dataset=precomputed,
                batch_size=args.batch_size,
                device=device,
                desc=f"classify {split} set",
            )

            results_file.parent.mkdir(exist_ok=True, parents=True)
            with results_file.open("w") as handle:
                handle.write(results.to_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="classify entity attributes as t/f")
    parser.add_argument("--editors", "-e", type=Path, help="path to editor experiment")
    parser.add_argument("--layers", "-l", nargs="+", type=int, help="layers to probe")
    parser.add_argument(
        "--model",
        "-m",
        choices=models.SUPPORTED_MODELS,
        default=models.GPT_J_NAME,
        help="model to classify on",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=data.SUPPORTED_DATASETS,
        default="counterfact",
        help="dataset to classify",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=editors.DEFAULT_BATCH_SIZE,
        help="model batch size",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16 model version")
    parser.add_argument("--device", help="device to run model on")
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
