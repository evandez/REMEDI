"""Evaluate editors on the CounterFact benchmark."""
import argparse
import json
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

from src import data, editors, metrics, models
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import Dataset

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def select_dedupe_flatten(dataset: Dataset, column: str) -> Dataset:
    """Select the given column and flatten it."""

    def _select_dedupe_flatten(row: dict) -> dict:
        prompts = list(
            {
                text
                for text in row[column]
                # Don't include prompts that were used to create the context.
                if text.lower() not in row["context"].lower()
            }
        )
        result = {"prompt": prompts}
        for key in data.ContextMediationSample.__required_keys__:
            result[key] = [row[key]] * len(prompts)
        return result

    column_names = dataset.column_names
    assert isinstance(column_names, list), column_names

    return dataset.map(
        _select_dedupe_flatten,
        batched=True,
        remove_columns=column_names,
        desc=f"select, dedupe, and flatten {column}",
    )


def group_by_id(results: editors.EditorEvaluateRun) -> OrderedDict:
    """Group results by sample ID."""
    grouped = defaultdict(list)
    for result in results.results:
        grouped[result.sample["id"]].append(result)
    return OrderedDict(grouped)


def main(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    experiment_utils.set_seed(args.seed)
    logging_utils.configure()
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16

    experiment_name = args.experiment_name or "benchmark"
    results_dir = experiment_utils.create_results_dir(
        experiment_name,
        root=args.results_dir,
        args=args,
        clear_if_exists=args.clear_results_dir,
    )

    editors_dir = args.editors
    if editors_dir is None:
        editors_dir = env_utils.determine_results_dir() / "editors"
    editors_dir /= "linear"
    logger.info(f"will look for editors in {editors_dir}")
    if not editors_dir.exists():
        raise ValueError(f"editors not found at {editors_dir}; maybe pass the -e flag")

    layers = args.layers
    if layers is None:
        layers = [str(layer_dir) for layer_dir in editors_dir.iterdir()]

    logger.info(f"loading {args.model} (device={device}, fp16={fp16})")
    mt = models.load_model(args.model, device=device, fp16=fp16)

    logger.info("loading several data sources")
    # TODO(evandez): Use full counterfact after splitting properly.
    dataset = data.load_dataset("counterfact", split="train[5000:10000]")
    attribute_snippets = data.load_attribute_snippets()
    tfidf_vectorizer = data.load_tfidf_vectorizer()
    paraphrase_prompts = select_dedupe_flatten(dataset, "paraphrase_prompts")
    generation_prompts = select_dedupe_flatten(dataset, "generation_prompts")

    for layer in layers:
        editor = editors.LinearEditor(mt=mt, layer=layer).to(device)
        weights_file = editors_dir / "linear" / str(layer) / "weights.pth"
        if not weights_file.exists():
            logger.warn(f"weights file for layer {layer} not found; skipping")
            continue
        logger.info(f"loaded layer {layer} editor from {weights_file}")
        state_dict = torch.load(weights_file, map_location=device)
        editor.load_state_dict(state_dict)

        results: dict[str, editors.EditorEvaluateRun] = {}
        for key, subset, kwargs in (
            ("prompts", dataset, dict(n_generate=1)),
            ("paraphrase_prompts", paraphrase_prompts, dict(n_generate=1)),
            (
                "generation_prompts",
                generation_prompts,
                dict(n_generate=args.n_generate),
            ),
        ):
            results_file = results_dir / f"{key}_results.json"
            if results_file.exists():
                logger.info(f"found existing {key} generations at {results_file}")
                with results_file.open("r") as handle:
                    results[key] = editors.EditorEvaluateRun.from_json(handle.read())
                continue

            results[key] = generations = editor.evaluate(
                subset,
                batch_size=args.batch_size,
                device=device,
                desc=f"{key} (layer {layer})",
                **kwargs,
            )

            logger.info(f"writing {key} generations to {results_file}")
            with results_file.open("w") as handle:
                handle.write(generations.to_json())

        prompts_results_by_id = group_by_id(results["prompts"])
        efficacy = metrics.efficacy(
            [
                [sample.after_target_mediated_score]
                for sample in prompts_results_by_id.values()
            ],
            [
                [sample.after_target_unmediated_score]
                for sample in prompts_results_by_id.values()
            ],
        )

        paraphrase_prompts_results_by_id = group_by_id(results["paraphrase_prompts"])
        paraphrase_efficacy = metrics.efficacy(
            [
                [sample.after_target_mediated_score for sample in samples]
                for samples in paraphrase_prompts_results_by_id.values()
            ],
            [
                [sample.after_target_unmediated_score for sample in samples]
                for samples in paraphrase_prompts_results_by_id.values()
            ],
        )

        generation_prompts_results_by_id = group_by_id(results["generation_prompts"])
        generation_prompts_outputs = [
            [sample.after_generations[0] for sample in samples]
            for samples in generation_prompts_results_by_id.values()
        ]

        consistency_references = []
        for samples in generation_prompts_results_by_id.values():
            sample = next(iter(samples))
            relation_id = sample.sample["requested_rewrite"]["relation_id"]
            target_id = sample.sample["requested_rewrite"]["target_new"]["target_id"]
            references = [
                snippet["text"]
                for snippet in attribute_snippets[relation_id][target_id]
            ]
            consistency_references.append(references)

        consistency = metrics.consistency(
            generation_prompts_outputs,
            consistency_references,
            tfidf_vectorizer,
        )

        fluency = metrics.fluency(generation_prompts_outputs)

        scores = {
            "efficacy": efficacy.to_dict(),
            "paraphrase_efficacy": paraphrase_efficacy.to_dict(),
            "consistency": consistency.to_dict(),
            "fluency": fluency.to_dict(),
        }
        scores_file = results_dir / "scores.json"
        with scores_file.open("r") as handle:
            json.dump(scores, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run full counterfact benchmark")
    parser.add_argument("--experiment-name", "-n", help="experiment name")
    parser.add_argument("--editors", "-e", type=Path, help="path to editor experiment")
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
        "--batch-size",
        type=int,
        default=editors.DEFAULT_BATCH_SIZE,
        help="model batch size",
    )
    parser.add_argument(
        "--n-generate",
        type=int,
        default=editors.DEFAULT_N_GENERATE,
        help="number of tokens to generate",
    )
    parser.add_argument("--results-dir", type=Path, help="write trained probes here")
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        help="clear old results and start anew",
    )
    parser.add_argument("--seed", type=int, default=123456, help="random seed")
    parser.add_argument("--fp16", action="store_true", help="use fp16 model version")
    parser.add_argument("--device", help="device to run model on")
    args = parser.parse_args()
    main(args)
