"""Train editors."""
import argparse
import logging
from typing import cast

from remedi import data, editors, models, precompute
from remedi.utils import experiment_utils, logging_utils
from remedi.utils.typing import Dataset

import torch

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Train the editors."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16
    use_entity = args.use_entity
    input_last_entity_token = edit_last_entity_token = not args.use_all_entity_tokens

    mt = models.load_model(args.model, device=device, fp16=fp16)

    dataset: Dataset = data.load_dataset(args.dataset, split="train[:5000]")
    dataset = precompute.from_args(args, dataset)

    layers = args.layers
    if layers is None:
        layers = models.determine_layers(mt)

    lam_u = args.lam_u
    if lam_u is None and args.dataset not in {"biosbias", "mcrae"}:
        lam_u = editors.DEFAULT_LAM_U

    dataset = data.maybe_train_test_split(dataset, test_size=args.hold_out)
    for editor_type in args.editor_types:
        editor_factory = editors.SUPPORTED_EDITORS[editor_type]
        editor_kwargs = dict(
            mt=mt,
            input_last_entity_token=input_last_entity_token,
            edit_last_entity_token=edit_last_entity_token,
        )
        if editor_type in {"mlp", "linear"}:
            editor_kwargs["use_entity"] = use_entity

        for layer in layers:
            logger.info(f"begin: editor={editor_type}, layer={layer}")

            editor: editors.Editor = editor_factory(layer=layer, **editor_kwargs)

            precomputed = cast(Dataset, dataset)
            if editor_type != "random":
                precomputed = precompute.editor_inputs_from_dataset(
                    mt=mt,
                    dataset=dataset,
                    layers=[layer],
                    device=device,
                    batch_size=args.batch_size,
                )

            editor.fit(
                dataset=precomputed,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lam_m=args.lam_m,
                lam_u=lam_u,
                lam_kl=args.lam_kl,
                lam_norm=args.lam_norm,
                lam_ess=args.lam_ess,
                device=device,
            )
            editors.save_editor(editor, editors_dir=experiment.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train one editor per layer")
    parser.add_argument(
        "--editor-types",
        "-t",
        nargs="+",
        choices=editors.SUPPORTED_EDITORS,
        default=("linear",),
        help="editor type to train",
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
        "--lam-m",
        type=float,
        default=editors.DEFAULT_LAM_M,
        help="p(mediated) term loss weight",
    )
    parser.add_argument(
        "--lam-u",
        type=float,
        help="1 - p(unmediated) term loss weight",
    )
    parser.add_argument(
        "--lam-norm",
        type=float,
        help="direction norm loss weight (not used by default)",
    )
    parser.add_argument(
        "--lam-ess",
        type=float,
        help="essence penalty loss weight (not used by default)",
    )
    parser.add_argument(
        "--hold-out",
        type=float,
        default=editors.DEFAULT_HOLD_OUT,
        help="held out fraction (if not already split)",
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
    data.add_dataset_args(parser)
    models.add_model_args(parser)
    precompute.add_preprocessing_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
