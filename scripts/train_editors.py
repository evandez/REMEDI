"""Train editors."""
import argparse
from pathlib import Path

from src import editors, precompute
from src.utils import dataset_utils, env, model_utils

import torch


def main(args: argparse.Namespace) -> None:
    """Train the editors."""
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = env.results_dir() / "editors" / args.editor_type
    results_dir.mkdir(exist_ok=True)

    mt = model_utils.load_model(args.mt, device=device, fp16=not args.no_fp16)
    dataset = dataset_utils.load_dataset(args.dataset)

    layers = args.layers
    if layers is None:
        layers = model_utils.determine_layers(mt)

    dataset = precompute.editor_inputs_from_dataset(
        mt=mt,
        dataset=dataset,
        layers=layers,
        device=device,
        batch_size=args.batch_size,
    )
    dataset = dataset_utils.maybe_train_test_split(dataset, test_size=args.hold_out)

    for layer in layers:
        assert args.editor_type == "linear", args.editor_type
        editor = editors.LinearEditor(mt=mt, layer=layer)
        editor.fit(
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            assume_inputs_precomputed=True,
        )
        torch.save(editor.state_dict(), results_dir / f"editor-l{layer}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train one editor per layer")
    parser.add_argument(
        "--editor-type",
        choices=("linear",),
        default="linear",
        help="editor type to train",
    )
    parser.add_argument("--model", default="gpt2-xl", help="model to edit")
    parser.add_argument("--dataset", default="counterfact", help="dataset to train on")
    parser.add_argument("--layers", type=int, nargs="+", help="layers to train for")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="training batch size"
    )
    parser.add_argument(
        "--hold-out",
        type=float,
        default=0.1,
        help="held out fraction (if not already split)",
    )
    parser.add_argument("--results-dir", type=Path, help="write trained probes here")
    parser.add_argument("--device", help="device to train on")
    parser.add_argument("--no-fp16", action="store_true", help="do not use fp16")
    args = parser.parse_args()
    main(args)
