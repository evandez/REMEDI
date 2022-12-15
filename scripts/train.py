"""Train editors."""
import argparse
import logging
from pathlib import Path

from src import data, editors, models, precompute
from src.utils import experiment_utils

import torch

EDITOR_FACTORIES = {
    "linear": editors.LinearEditor,
    "biaffine": editors.BiaffineEditor,
    "mlp": editors.MlpEditor,
    "random": editors.RandomEditor,
}

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Train the editors."""
    experiment_utils.set_seed(args.seed)
    data.disable_caching()

    experiment_name = args.experiment_name or "editors"
    results_dir = experiment_utils.create_results_dir(
        experiment_name,
        root=args.results_dir,
        args=args,
        clear_if_exists=args.clear_results_dir,
    )

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16
    use_entity = args.use_entity
    input_last_entity_token = edit_last_entity_token = not args.use_all_entity_tokens

    logger.info(f"loading {args.model} (device={device}, fp16={fp16})")
    mt = models.load_model(args.model, device=device, fp16=fp16)

    dataset = data.load_dataset(args.dataset, split="train[:5000]")

    layers = args.layers
    if layers is None:
        layers = models.determine_layers(mt)

    dataset = data.maybe_train_test_split(dataset, test_size=args.hold_out)
    for editor_type in args.editor_types:
        editor_factory = EDITOR_FACTORIES[editor_type]
        editor_kwargs = dict(
            mt=mt,
            input_last_entity_token=input_last_entity_token,
            edit_last_entity_token=edit_last_entity_token,
        )
        if editor_type in {"mlp", "linear"}:
            editor_kwargs["use_entity"] = use_entity

        for layer in layers:
            logger.info(f"begin: editor={editor_type}, layer={layer}")

            editor_results_dir = results_dir / editor_type / str(layer)
            editor_results_dir.mkdir(exist_ok=True, parents=True)

            editor: editors.Editor = editor_factory(layer=layer, **editor_kwargs)

            precomputed = precompute.editor_inputs_from_dataset(
                mt=mt,
                dataset=dataset,
                layers=[layer],
                device=device,
                batch_size=args.batch_size,
            )

            editor_file = editor_results_dir / f"weights.pth"
            if editor_file.exists():
                logger.info(f"found existing editor at {editor_file}")
                state_dict = torch.load(editor_file, map_location=device)
                editor.load_state_dict(state_dict)
            else:
                editor.fit(
                    dataset=precomputed,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lam_kl=args.lam_kl,
                    lam_adv=args.lam_adv,
                    device=device,
                )
                logger.info(f"saving editor to {editor_file}")
                torch.save(editor.state_dict(), editor_file)

            for split in ("train", "test"):
                eval_file = editor_results_dir / f"{split}-eval.json"
                if eval_file.exists() and not args.rerun_eval:
                    logger.info(f"found existing {split} eval results at {eval_file}")
                    continue

                results = editor.evaluate(
                    precomputed[split],  # type: ignore
                    batch_size=args.batch_size,
                    device=device,
                    alpha=args.eval_alpha,
                    n_top=args.eval_n_top,
                    n_generate=args.eval_n_generate,
                )
                logger.info(f"saving {split} eval to {eval_file}")
                with eval_file.open("w") as handle:
                    handle.write(results.to_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train one editor per layer")
    parser.add_argument("--experiment-name", "-n", help="experiment name")
    parser.add_argument(
        "--editor-types",
        "-e",
        nargs="+",
        choices=EDITOR_FACTORIES.keys(),
        default=("linear",),
        help="editor type to train",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=models.SUPPORTED_MODELS,
        default="gpt2-xl",
        help="model to edit",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=data.SUPPORTED_DATASETS,
        default="counterfact",
        help="dataset to train on",
    )
    parser.add_argument(
        "--layers", "-l", type=int, nargs="+", help="layers to train for"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=editors.DEFAULT_MAX_EPOCHS,
        help="max training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=editors.DEFAULT_BATCH_SIZE,
        help="training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=editors.DEFAULT_LR, help="learning rate"
    )
    parser.add_argument(
        "--lam-kl",
        type=float,
        default=editors.DEFAULT_LAM_KL,
        help="kl div loss weight",
    )
    parser.add_argument(
        "--lam-adv",
        type=float,
        default=editors.DEFAULT_LAM_ADV,
        help="adversarial term loss weight",
    )
    parser.add_argument(
        "--hold-out",
        type=float,
        default=editors.DEFAULT_HOLD_OUT,
        help="held out fraction (if not already split)",
    )
    parser.add_argument(
        "--eval-alpha",
        type=float,
        default=editors.DEFAULT_ALPHA,
        help="step size for adding direction in eval",
    )
    parser.add_argument(
        "--eval-n-top",
        type=int,
        default=editors.DEFAULT_N_TOP,
        help="number of top tokens/scores to report in eval",
    )
    parser.add_argument(
        "--eval-n-generate",
        type=int,
        default=editors.DEFAULT_N_GENERATE,
        help="number of tokens to generate in eval",
    )
    parser.add_argument("--results-dir", type=Path, help="write trained probes here")
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        help="clear old results and start anew",
    )
    parser.add_argument(
        "--use-entity",
        action="store_true",
        help="use entity in linear/mlp editors",
    )
    parser.add_argument(
        "--use-all-entity-tokens",
        action="store_true",
        help="edit all entity tokens instead of just last",
    )
    parser.add_argument("--rerun-eval", action="store_true", help="rerun eval step")
    parser.add_argument("--device", help="device to train on")
    parser.add_argument("--seed", type=int, default=123456, help="random seed")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    args = parser.parse_args()
    main(args)