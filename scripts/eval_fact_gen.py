"""Evaluate editors on the Counterfact benchmark."""
import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import cast

from remedi import benchmarks, data, editors, models, precompute
from remedi.utils import experiment_utils, logging_utils
from remedi.utils.typing import Dataset, Device

import torch
import torch.utils.data
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

BENCHMARKS = (
    "efficacy",
    "paraphrase",
    "generation",
    "essence",
)


def _prefix_context(sample: dict) -> dict:
    """Prepend context to all prompts used in the eval."""
    entity = sample["entity"]
    prompt = sample["prompt"]
    context = sample["context"]

    prompt_in_context = precompute.prompt_in_context_from_sample(
        entity, prompt, context
    )

    source = {**sample["source"]}
    for key in ("generation_prompts", "paraphrase_prompts"):
        source[key] = [
            precompute.prompt_in_context_from_sample(entity, other_prompt, context)
            for other_prompt in source[key]
        ]

    return {"source": source, "prompt": prompt_in_context}


def _prefix_essence_prompt_template(sample: dict) -> str:
    """Prompt template for essence benchmark of prefix baseline."""
    entity = sample["entity"]
    context = sample["context"]
    prompt = benchmarks.DEFAULT_PROMPT_TEMPLATE.format(entity)
    prompt_in_context = precompute.prompt_in_context_from_sample(
        entity, prompt, context
    )
    return prompt_in_context


def _prefix_essence_post_process(generation: str) -> str:
    """Post-process essence generation for prefix baseline."""
    return ". ".join(generation.split(". ")[1:])


def _replace_entity(attribute_snippets: data.AttributeSnippets, sample: dict) -> dict:
    """Replace entity with one that has same target attribute."""
    requested_rewrite = sample["source"]["requested_rewrite"]
    relation_id = requested_rewrite["relation_id"]
    target_id = requested_rewrite["target_new"]["id"]
    candidates = attribute_snippets[relation_id][target_id]
    replacement = random.choice(candidates)["name"]

    entity = sample["entity"]
    context = sample["context"]
    prompt = sample["prompt"]

    source = {**sample["source"]}
    for key in ("generation_prompts", "paraphrase_prompts"):
        source[key] = [x.replace(entity, replacement) for x in source[key]]

    return {
        "entity": replacement,
        "context": context.replace(entity, replacement),
        "prompt": prompt.replace(entity, replacement),
        "source": source,
    }


@torch.inference_mode()
def _precompute_essence_references(
    mt: models.ModelAndTokenizer, dataset: Dataset, device: Device | None = None
) -> list[list[str]]:
    """Precompute essence references to save some compute."""
    prompts = [
        benchmarks.DEFAULT_PROMPT_PREFIX
        + benchmarks.DEFAULT_PROMPT_TEMPLATE.format(x["entity"])
        for x in dataset
    ]
    loader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, prompts),
        batch_size=editors.DEFAULT_BATCH_SIZE,
    )
    references = []
    for batch in tqdm(loader, desc="precompute essence refs"):
        with models.set_padding_side(mt, padding_side="left"):
            inputs, _ = precompute.inputs_from_batch(mt, batch, device=device)
        outputs = mt.model.generate(
            **inputs,
            max_length=benchmarks.DEFAULT_MAX_LENGTH,
            do_sample=True,
            top_k=benchmarks.DEFAULT_TOP_K_SAMPLING,
            pad_token_id=mt.tokenizer.eos_token_id,
        )
        references += [
            [r[len(benchmarks.DEFAULT_PROMPT_PREFIX) :]]
            for r in mt.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
    return references


def main(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    logger.info("loading several data sources")
    if args.small:
        split = "train[5000:6000]"
    else:
        split = "train[5000:10000]"
    dataset = data.load_dataset("counterfact", split=split)
    dataset = precompute.from_args(args, dataset)
    attribute_snippets = data.load_attribute_snippets()
    tfidf_vectorizer = data.load_counterfact_tfidf_vectorizer()

    essence_references = None
    if "essence" in args.benchmarks:
        essence_refs_file = experiment.results_dir / "essence_references.json"
        if essence_refs_file.exists():
            logger.info(f"found essence refs at {essence_refs_file}")
            with essence_refs_file.open("r") as handle:
                essence_references = [[l] for l in json.load(handle)["references"]]
        else:
            essence_references = _precompute_essence_references(
                mt, dataset, device=device
            )

            logger.info(f"saving precomputed references to {essence_refs_file}")
            essence_refs_file.parent.mkdir(exist_ok=True, parents=True)
            with essence_refs_file.open("w") as handle:
                json.dump(
                    {"references": [rs[0] for rs in essence_references]},
                    handle,
                )

    baseline = args.baseline
    if baseline is not None:
        for banned in ("layers", "editors_dir"):
            if getattr(args, banned, None) is not None:
                raise ValueError(f"cannot set --{banned} with --baseline")

        dataset = precompute.editor_inputs_from_dataset(
            dataset=dataset,
            mt=mt,
            return_entity_hiddens=False,
            return_attribute_hiddens=False,
            return_token_ranges=False,
            return_target_token_ids=True,
            desc="precompute target token ids",
        )

        if baseline == "prefix":
            dataset = dataset.map(_prefix_context, desc="prefix context")
        elif baseline == "replace":
            dataset = dataset.map(
                partial(_replace_entity, attribute_snippets), desc="replace entities"
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
                results_file = (
                    experiment.results_dir / baseline / f"{benchmark_name}.json"
                )
            else:
                results_file = (
                    experiment.results_dir
                    / editor_type
                    / str(layer)
                    / f"{benchmark_name}.json"
                )
            if results_file.exists():
                logger.info(
                    f"found existing {benchmark_name} results for layer {layer} "
                    f"at {results_file}"
                )
                continue

            essence_kwargs: dict = {}
            if baseline == "prefix":
                essence_kwargs["prompt_template"] = _prefix_essence_prompt_template
                essence_kwargs["post_process"] = _prefix_essence_post_process

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
                    tfidf_vectorizer=tfidf_vectorizer,
                    use_references=essence_references,
                    **benchmark_kwargs,
                    **essence_kwargs,
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

            metrics_file = results_file.parent / f"{benchmark_name}_metrics.json"
            with metrics_file.open("w") as handle:
                json.dump(results.metrics.to_dict(), handle)


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
        choices=("prefix", "replace"),
        help="run a baseline instead of evaluating an editor",
    )
    parser.add_argument(
        "--small", action="store_true", help="run on a small subset of data"
    )
    # No dataset args because this only works for counterfact
    models.add_model_args(parser)
    precompute.add_preprocessing_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
