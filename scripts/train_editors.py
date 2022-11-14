"""Train editors."""
import argparse
import shutil
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
    if args.clear_results_dir and results_dir.exists():
        print(f"clearing results dir {results_dir}")
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    mt = model_utils.load_model(args.model, device=device, fp16=not args.no_fp16)
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

        print(f"---- Layer {layer} ----")
        editor = editors.LinearEditor(mt=mt, layer=layer)

        editor_file = results_dir / f"editor-l{layer}.pth"
        if editor_file.exists():
            print(f"found existing editor at {editor_file}")
            state_dict = torch.load(editor_file, map_location=device)
            editor.load_state_dict(state_dict)
        else:
            editor.fit(dataset=dataset, batch_size=args.batch_size, device=device)
            print(f"saving editor to {editor_file}")
            torch.save(editor.state_dict(), editor_file)

        eval_file = results_dir / f"editor-l{layer}-eval.json"
        if eval_file.exists():
            print(f"found existing eval results at {eval_file}")
            continue

        eval_run = editor.evaluate(
            dataset["test"],
            batch_size=args.batch_size,
            device=device,
            n_top=args.eval_n_top,
            n_generate=args.eval_n_generate,
        )
        with eval_file.open("w") as handle:
            handle.write(eval_run.to_json())


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
    parser.add_argument(
        "--eval-n-top",
        type=int,
        default=10,
        help="number of top words/scores to report in eval",
    )
    parser.add_argument(
        "--eval-n-generate",
        type=int,
        default=10,
        help="number of words to generate in eval",
    )
    parser.add_argument("--results-dir", type=Path, help="write trained probes here")
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        help="clear old results and start anew",
    )
    parser.add_argument("--device", help="device to train on")
    parser.add_argument("--no-fp16", action="store_true", help="do not use fp16")
    args = parser.parse_args()
    main(args)
