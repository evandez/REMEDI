"""Standalone functions for benchmarking editor performance across metrics."""
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence, cast

from src import data, editors, metrics, models, precompute
from src.utils.typing import Dataset, Device, StrSequence

import torch
import torch.utils.data
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PREFIX = "The following is an except from a Wikipedia article:\n\n"
DEFAULT_PROMPT_TEMPLATE = "{} is"
DEFAULT_MAX_LENGTH = 100
DEFAULT_TOP_K = 5
DEFAULT_N_TOP_TOKENS = DEFAULT_TOP_K


@dataclass(frozen=True)
class EssenceSample(DataClassJsonMixin):
    """Single sample from the essence benchmark.

    Fields:
        id: ID of the sample.
        generation: The generated text from the prompt.
        references: The reference texts.
        essence: The essence score for this example.
        fluency_generation: Fluency score for the generation.
        fluency_references: Fluency score for the references.

    """

    id: str

    generation: str
    references: list[str]

    essence_score: float
    fluency_generation_score: float
    fluency_references_score: float


@dataclass(frozen=True)
class EssenceMetrics(DataClassJsonMixin):
    """Wrapper around essence benchmark metrics.

    Fields:
        essence: TF-IDF similarity to references.
        fluency_generation: Average n-gram entropy of generations.
        fluency_references: Average n-gram entropy of references.

    """

    essence: metrics.Metric
    fluency_generaton: metrics.Metric
    fluency_references: metrics.Metric


@dataclass(frozen=True)
class EssenceBenchmarkResults(DataClassJsonMixin):
    """Essence benchmark results."""

    samples: list[EssenceSample]
    metrics: EssenceMetrics


PromptTemplateFn = Callable[[dict], str]
GenerationPostProcessFn = Callable[[str], str]


@torch.inference_mode()
def essence(
    *,
    dataset: Dataset,
    editor: editors.Editor | None = None,
    mt: models.ModelAndTokenizer | None = None,
    alpha: float = editors.DEFAULT_ALPHA,
    beta: float = editors.DEFAULT_BETA,
    batch_size: int = editors.DEFAULT_BATCH_SIZE,
    prompt_prefix: str | None = DEFAULT_PROMPT_PREFIX,
    prompt_template: str | PromptTemplateFn = DEFAULT_PROMPT_TEMPLATE,
    post_process: GenerationPostProcessFn | None = None,
    reference_prompt_prefix: str | None = DEFAULT_PROMPT_PREFIX,
    reference_prompt_template: str | PromptTemplateFn = DEFAULT_PROMPT_TEMPLATE,
    reference_post_process: GenerationPostProcessFn | None = None,
    max_new_tokens: int | None = None,
    max_length: int | None = None,
    use_references: Sequence[StrSequence] | None = None,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    desc: str | None = None,
    device: Device | None = None,
) -> EssenceBenchmarkResults:
    """Measures how well the editor preserves the edited entity's essence."""
    if editor is None and mt is None:
        raise ValueError("must set at least one of `editor` and `mt`")
    if isinstance(prompt_template, str) and prompt_template.count("{}") != 1:
        raise ValueError(f"prompt template needs 1 empty slot: {prompt_template}")
    if use_references is not None and len(use_references) != len(dataset):
        raise ValueError(
            "size mismatch: "
            f"use_references={len(use_references)}, dataset={len(dataset)}"
        )

    if mt is None:
        assert editor is not None
        mt = editor.mt
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH
    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_tfidf_vectorizer()
    if desc is None:
        desc = "essence benchmark"

    generate_kwargs = dict(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        pad_token_id=mt.tokenizer.eos_token_id,
    )

    generations = []
    reference_groups: list[list[str]] = []
    essence_scores = []
    fluency_generation_scores = []
    fluency_references_scores = []
    with dataset.formatted_as("torch"):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset), batch_size=batch_size
        )
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{desc} [generate]")):
            ids = batch["id"]
            entities = batch["entity"]
            attributes = batch["attribute"]

            # Step 1: If needed, generate reference texts.
            if use_references is None:
                reference_prompts = _create_essence_prompts(
                    batch,
                    prompt_template=reference_prompt_template,
                    prompt_prefix=reference_prompt_prefix,
                )
                with models.set_padding_side(mt, padding_side="left"):
                    reference_inputs, _ = precompute.inputs_from_batch(
                        mt, reference_prompts, device=device
                    )
                reference_outputs = mt.model.generate(
                    **reference_inputs, **generate_kwargs
                )
                batch_reference_groups = [
                    [r]
                    for r in mt.tokenizer.batch_decode(
                        reference_outputs, skip_special_tokens=True
                    )
                ]
                if reference_prompt_prefix is not None:
                    batch_reference_groups = [
                        [r[len(reference_prompt_prefix) :]]
                        for [r] in batch_reference_groups
                    ]
                if reference_post_process is not None:
                    batch_reference_groups = [
                        [reference_post_process(r)] for [r] in batch_reference_groups
                    ]
            else:
                start = batch_index * batch_size
                end = start + len(entities)
                batch_reference_groups = [list(rs) for rs in use_references[start:end]]
            reference_groups += batch_reference_groups

            # Step 2: Generate post-edit text.
            prompts = _create_essence_prompts(
                batch, prompt_template=prompt_template, prompt_prefix=prompt_prefix
            )
            with models.set_padding_side(mt, padding_side="left"):
                inputs, _ = precompute.inputs_from_batch(mt, prompts, device=device)
            if editor is not None:
                with editors.apply(
                    editor, alpha=alpha, beta=beta, device=device
                ) as edited_mt:
                    outputs = edited_mt.model.generate(
                        data.ContextMediationBatch(
                            id=ids,
                            source=batch["source"],
                            entity=entities,
                            prompt=prompts,
                            attribute=attributes,
                            context=batch["context"],
                            target_mediated=None,
                            target_unmediated=None,
                        ),
                        inputs=inputs,
                        padding_side="left",
                        **generate_kwargs,
                    )
            else:
                outputs = mt.model.generate(**inputs, **generate_kwargs)

            batch_generations = mt.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            if prompt_prefix is not None:
                batch_generations = [g[len(prompt_prefix) :] for g in batch_generations]
            if post_process is not None:
                batch_generations = [post_process(g) for g in batch_generations]
            generations += batch_generations

            for (sid, entity, attribute, generation, references) in zip(
                ids,
                entities,
                attributes,
                batch_generations,
                batch_reference_groups,
            ):
                logger.debug(f"ID={sid} ENTITY={entity}, ATTR={attribute}")
                logger.debug(f"ID={sid} REFERENCES={references}")
                logger.debug(f"ID={sid} GENERATION={generation}")

                essence_score = metrics.tfidf_similarity(
                    generation, references, tfidf_vectorizer
                )
                essence_scores.append(essence_score)

                fluency_generation_score = metrics.weighted_n_gram_entropy(generation)
                fluency_generation_scores.append(fluency_generation_score)

                fluency_references_score = metrics.weighted_n_gram_entropy(references)
                fluency_references_scores.append(fluency_references_score)

    samples = [
        EssenceSample(
            id=sample["id"],
            generation=generation,
            references=references,
            essence_score=essence_score,
            fluency_generation_score=fluency_generation_score,
            fluency_references_score=fluency_references_score,
        )
        for sample, generation, references, essence_score, fluency_generation_score, fluency_references_score in zip(
            dataset,
            generations,
            reference_groups,
            essence_scores,
            fluency_generation_scores,
            fluency_references_scores,
        )
    ]

    metrics_kwargs = {}
    for key, scores in (
        ("essence", essence_scores),
        ("fluency_generaton", fluency_generation_scores),
        ("fluency_references", fluency_references_scores),
    ):
        metric = metrics.Metric.aggregate(scores, store_values=False)
        logger.info(f"{key} mean={metric.mean:.2f}, std={metric.std:.2f}")
        metrics_kwargs[key] = metric

    return EssenceBenchmarkResults(
        samples=samples,
        metrics=EssenceMetrics(**metrics_kwargs),
    )


def _create_essence_prompts(
    batch: dict,
    prompt_template: str | PromptTemplateFn = DEFAULT_PROMPT_TEMPLATE,
    prompt_prefix: str | None = DEFAULT_PROMPT_PREFIX,
) -> StrSequence:
    """Create list of fully formatted prompts."""
    entities = batch["entity"]
    if isinstance(prompt_template, str):
        prompts = [prompt_template.format(entity) for entity in entities]
    else:
        prompts = [
            prompt_template(
                {
                    key: batch[key][bi]
                    for key in data.ContextMediationSample.__required_keys__
                    if key != "source"
                }
            )
            for bi in range(len(entities))
        ]

    if prompt_prefix is not None:
        prompts = [prompt_prefix + prompt for prompt in prompts]

    return prompts


@dataclass(frozen=True)
class ClassifierOutputs(DataClassJsonMixin):
    """Wrapper around a single classifier sample.

    Fields:
        logp_target: Log probability of target word under the LM.
        logp_comparator: Ditto, for comparator word.
        score_target: Score that classifier assigns to target.
        score_comparator: Score that classifier assigns to comparator.

    """

    logp_target: float
    logp_comparator: float
    score_target: float
    score_comparator: float

    @property
    def label(self) -> bool:
        """Get the true label value for this sample."""
        return self.logp_target > self.logp_comparator

    @property
    def prediction(self) -> bool:
        """Get the predicted label for this sample."""
        return self.score_target > self.score_comparator

    @property
    def correct(self) -> bool:
        """Get whether the prediction was correct or not."""
        return self.prediction == self.label


@dataclass(frozen=True)
class ClassificationSample(DataClassJsonMixin):
    """Wrapper around a single classification sample."""

    id: str
    contextual: ClassifierOutputs
    decontextual: ClassifierOutputs


@dataclass(frozen=True)
class ClassifierMetrics(DataClassJsonMixin):
    """Wrapper around all classification scores."""

    f1: float
    mcc: float
    accuracy: float


@dataclass(frozen=True)
class ClassifierResults(DataClassJsonMixin):
    """Wraps results for a specific classifier."""

    contextual: ClassifierMetrics
    decontextual: ClassifierMetrics


@dataclass(frozen=True)
class ClassificationBenchmarkResults(DataClassJsonMixin):
    """Classification benchmark results.

    Fields:
        samples: Individual results for each sample in dataset.
        editor: Metrics from using editor as classifier.
        baseline: Metrics from always guessing majority label.

    """

    samples: list[ClassificationSample]
    editor: ClassifierResults
    baseline: ClassifierResults


@torch.inference_mode()
def classification(
    *,
    editor: editors.Editor,
    dataset: Dataset,
    batch_size: int = editors.DEFAULT_BATCH_SIZE,
    entity_layer: int | None = None,
    desc: str | None = None,
    device: Device | None = None,
    **kwargs: Any,
) -> ClassificationBenchmarkResults:
    """Measure how well the editor acts as a classifier.

    Args:
        editor: The editor to benchmark.
        dataset: The dataset to benchmark on.
        batch_size: Max number of samples to process at once.
        entity_layer: The layer to get the entity rep from. This can be different
            from the edit layer!
        layer: The layer to grab entity reps from.
        desc: A tqdm description.
        device: Send model and inputs to device.

    Returns:
        The benchmark results.

    """
    if desc is None:
        desc = "classification benchmark"

    if entity_layer is None:
        entity_layer = editor.layer
    layers = sorted({editor.layer, entity_layer})

    precomputed = precompute.classification_inputs_from_dataset(
        editor.mt,
        dataset,
        layers=layers,
        batch_size=batch_size,
        device=device,
        desc=f"{desc} [compute reps]",
    )

    runs: dict[str, list[editors.EditorClassificationResult]] = {}
    for key in ("contextual", "decontextual"):
        runs[key] = editor.classify(
            dataset=precomputed,
            take_entity_from="prompt_in_context" if key == "contextual" else "prompt",
            batch_size=batch_size,
            entity_layer=entity_layer,
            device=device,
            desc=f"{desc} [classify {key}]",
            **kwargs,
        ).results

    samples = []
    for pre, rc, rd in zip(precomputed, runs["contextual"], runs["decontextual"]):
        sample = ClassificationSample(
            id=rc.sample["id"],
            contextual=ClassifierOutputs(
                logp_target=pre["prompt_in_context.target.logp"],
                logp_comparator=pre["prompt_in_context.comparator.logp"],
                score_target=rc.score_mediated,
                score_comparator=rc.score_unmediated,
            ),
            decontextual=ClassifierOutputs(
                logp_target=pre["prompt.target.logp"],
                logp_comparator=pre["prompt.comparator.logp"],
                score_target=rd.score_unmediated,
                score_comparator=rd.score_mediated,
            ),
        )
        samples.append(sample)

    benchmark_results_kwargs: dict = defaultdict(dict)
    for method in ("editor", "baseline"):
        for task in ("contextual", "decontextual"):
            y_true = [getattr(sample, key).label for sample in samples]
            y_pred = [getattr(sample, key).prediction for sample in samples]

            # In the baseline case, pick the majority label for everything.
            if method == "baseline":
                majority = sum(y_true) >= len(y_true) / 2
                y_pred = [majority] * len(y_true)

            # In the contextual case, we want to classify whether the model will *not* make
            # the correct prediction. This does not change accuracy/mcc but does change f1.
            if task == "contextual":
                y_true = [not x for x in y_true]
                y_pred = [not x for x in y_pred]

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            benchmark_results_kwargs[method][task] = ClassifierMetrics(
                f1=f1, mcc=mcc, accuracy=accuracy
            )
    benchmark_results_kwargs = {
        key: ClassifierResults(**classifier_results_kwargs)
        for key, classifier_results_kwargs in benchmark_results_kwargs.items()
    }

    return ClassificationBenchmarkResults(samples=samples, **benchmark_results_kwargs)


@dataclass(frozen=True)
class EfficacySample(DataClassJsonMixin):
    """Wrapper around a single efficacy sample."""

    id: str
    prompt: str
    target_score: float
    comparator_score: float


@dataclass(frozen=True)
class EfficacyBenchmarkResults(DataClassJsonMixin):
    """Wrapper around efficacy benchmark results."""

    samples: list[EfficacySample]
    metrics: metrics.EfficacyMetrics


@torch.inference_mode()
def efficacy(
    *,
    dataset: Dataset,
    editor: editors.Editor | None = None,
    mt: models.ModelAndTokenizer | None = None,
    desc: str | None = None,
    **kwargs: Any,
) -> EfficacyBenchmarkResults:
    """Run the efficacy benchmark.

    Uses the "prompt" column in the dataset to generate the next token before and
    after editing, then compares the probabilities between the mediated and unmediated
    tokens.
    """
    if (editor is None) == (mt is None):
        raise ValueError("must one of `editor` and `mt`, not both")

    if desc is None:
        desc = "efficacy benchmark"

    evaluate_kwargs = dict(max_new_tokens=1, desc=desc, **kwargs)
    if editor is None:
        assert mt is not None
        run = editors.NullEditor(mt=mt, layer=0).evaluate(
            dataset, return_after=False, **evaluate_kwargs
        )
        target_score_key = "before_target_mediated_score"
        comparator_score_key = "before_target_unmediated_score"
    else:
        assert mt is None
        run = editor.evaluate(dataset, return_before=False, **evaluate_kwargs)
        target_score_key = "after_target_mediated_score"
        comparator_score_key = "after_target_unmediated_score"

    samples = []
    for result in run.results:
        sid = result.sample["id"]
        prompt = result.sample["prompt"]

        target_score = getattr(result, target_score_key)
        assert target_score is not None

        comparator_score = getattr(result, comparator_score_key)
        assert comparator_score is not None

        logger.debug(f"ID={sid} SCORE_T={target_score} SCORE_COMP={comparator_score}")
        sample = EfficacySample(
            id=sid,
            prompt=prompt,
            target_score=target_score,
            comparator_score=comparator_score,
        )
        samples.append(sample)

    efficacy_metrics = metrics.efficacy(
        [[sample.target_score] for sample in samples],
        [[sample.comparator_score] for sample in samples],
        store_values=False,
    )
    return EfficacyBenchmarkResults(samples=samples, metrics=efficacy_metrics)


@dataclass(frozen=True)
class ParaphraseSample(DataClassJsonMixin):
    """Wrapper around a single paraphrase benchmark sample."""

    id: str
    prompts: list[EfficacySample]
    efficacy_score: float
    efficacy_magnitude: float


@dataclass(frozen=True)
class CounterFactParaphraseBenchmarkResults(DataClassJsonMixin):
    """Wrapper around paraphrase benchmark results."""

    samples: list[ParaphraseSample]
    metrics: metrics.EfficacyMetrics


@torch.inference_mode()
def counterfact_paraphrase(
    *,
    editor: editors.Editor | None = None,
    mt: models.ModelAndTokenizer | None = None,
    dataset: Dataset,
    desc: str | None = None,
    **kwargs: Any,
) -> CounterFactParaphraseBenchmarkResults:
    """Run the CounterFact paraphrase benchmark.

    Since this benchmark relies on extra data, it can only be used with the CounterFact
    dataset. The `counterfact_generation` benchmark is like this as well.

    This function expects that each sample in the dataset supports an access like:

        prompts = sample["source"]["generation_prompts"]

    """
    if desc is None:
        desc = "paraphrase benchmark"
    dataset = _counterfact_select_and_flatten(
        dataset, "paraphrase_prompts", desc=f"{desc} [flatten dataset]"
    )
    efficacy_benchmark = efficacy(
        editor=editor,
        mt=mt,
        dataset=dataset,
        desc=desc,
        **kwargs,
    )

    results_by_sample_id: dict = defaultdict(list)
    for result in efficacy_benchmark.samples:
        results_by_sample_id[result.id].append(result)
    results_by_sample_id = OrderedDict(results_by_sample_id)

    efficacy_metrics = metrics.efficacy(
        [
            [result.target_score for result in results]
            for results in results_by_sample_id.values()
        ],
        [
            [result.comparator_score for result in results]
            for results in results_by_sample_id.values()
        ],
    )

    # Reformat EfficacySample -> ParaphraseSample
    samples = []
    for (sid, results), efficacy_score, efficacy_magnitude in zip(
        results_by_sample_id.items(),
        cast(list, efficacy_metrics.score.values),
        cast(list, efficacy_metrics.magnitude.values),
    ):
        sample = ParaphraseSample(
            id=sid,
            prompts=results,
            efficacy_score=efficacy_score,
            efficacy_magnitude=efficacy_magnitude,
        )
        samples.append(sample)

    return CounterFactParaphraseBenchmarkResults(
        samples=samples,
        metrics=efficacy_metrics.without_values(),
    )


@dataclass(frozen=True)
class GenerationSample(DataClassJsonMixin):
    """Wrapper around a single sample from the generation benchmark."""

    id: str
    generations: list[str]
    references: list[str]
    fluency_score: float
    consistency_score: float


@dataclass(frozen=True)
class GenerationMetrics(DataClassJsonMixin):
    """Wrapper around all generation metrics."""

    fluency: metrics.Metric
    consistency: metrics.Metric


@dataclass(frozen=True)
class CounterFactGenerationBenchmarkResults(DataClassJsonMixin):
    """Wrapper around generation benchmark results."""

    samples: list[GenerationSample]
    metrics: GenerationMetrics


@torch.inference_mode()
def counterfact_generation(
    *,
    dataset: Dataset,
    editor: editors.Editor | None = None,
    mt: models.ModelAndTokenizer | None = None,
    attribute_snippets: data.AttributeSnippets | None = None,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str | None = None,
    **kwargs: Any,
) -> CounterFactGenerationBenchmarkResults:
    """Run the CounterFact generation benchmark.

    Free-form generates on several "generation prompts" per sample, and records
    the fluency of the generations (measured by weighted n-gram entropy) and
    consistency with other texts about entities with the same attribute.

    This benchmark *requires* the dataset to be CounterFact or something that looks
    like it, since it uses extra data that is specific to CounterFact.

    Specifically, it expects each sample can be accessed like:

        prompts = sample["source"]["generation_prompts"]

    """
    if (mt is None) == (editor is None):
        raise ValueError("must set one of `editor` or `mt`, not both")

    if attribute_snippets is None:
        attribute_snippets = data.load_attribute_snippets()
    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_tfidf_vectorizer()
    if max_new_tokens is None and max_length is None:
        max_length = DEFAULT_MAX_LENGTH
    if desc is None:
        desc = "generate benchmark"

    dataset = _counterfact_select_and_flatten(
        dataset, "generation_prompts", desc=f"{desc} [flatten dataset]"
    )

    evaluate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        desc=f"{desc} [run model]",
        **kwargs,
    )
    if editor is None:
        assert mt is not None
        run = editors.NullEditor(mt=mt, layer=0).evaluate(
            dataset, return_after=False, **evaluate_kwargs
        )
        generations_key = "before_generations"
    else:
        assert mt is None
        run = editor.evaluate(dataset, return_before=False, **evaluate_kwargs)
        generations_key = "after_generations"
    run_results_by_id = _group_results_by_id(run)

    samples = []
    for sid, results in tqdm(run_results_by_id.items(), desc=f"{desc} [tfidf]"):
        result = next(iter(results))
        cf_requested_rewrite = result.sample["source"]["requested_rewrite"]
        relation_id = cf_requested_rewrite["relation_id"]
        target_id = cf_requested_rewrite["target_new"]["id"]

        generations = [getattr(result, generations_key)[0] for result in results]
        references = [
            snippet["text"] for snippet in attribute_snippets[relation_id][target_id]
        ]

        consistency_score = metrics.tfidf_similarity(
            generations, references, tfidf_vectorizer
        )
        fluency_score = metrics.weighted_n_gram_entropy(generations)

        entity = result.sample["entity"]
        attribute = result.sample["attribute"]
        logger.debug(f"ID={sid} ENTITY={entity}, ATTR={attribute}")
        logger.debug(f"ID={sid} REFERENCES={references}")
        logger.debug(f"ID={sid} GENERATIONS={generations}")

        sample = GenerationSample(
            id=sid,
            generations=generations,
            references=references,
            fluency_score=fluency_score,
            consistency_score=consistency_score,
        )
        samples.append(sample)

    fluency = metrics.Metric.aggregate(
        [sample.fluency_score for sample in samples], store_values=False
    )
    consistency = metrics.Metric.aggregate(
        [sample.consistency_score for sample in samples], store_values=False
    )
    generation_metrics = GenerationMetrics(fluency=fluency, consistency=consistency)
    return CounterFactGenerationBenchmarkResults(
        samples=samples, metrics=generation_metrics
    )


def _counterfact_select_and_flatten(
    dataset: Dataset, column: str, desc: str | None = None
) -> Dataset:
    """Select the given column in counterfact, dedupe it, and flatten it."""
    column_names = data.column_names(dataset)

    def select_and_flatten_counterfact_row(row: dict) -> dict:
        prompts = list(set(row["source"][0][column]))
        result = {"prompt": prompts}
        for key in data.ContextMediationSample.__required_keys__:
            if key not in result:
                result[key] = [row[key][0]] * len(prompts)
        return result

    return dataset.map(
        select_and_flatten_counterfact_row,
        batched=True,
        batch_size=1,
        remove_columns=column_names,
        desc=desc,
    )


def _group_results_by_id(results: editors.EditorEvaluateRun) -> OrderedDict:
    """Group results by sample ID."""
    grouped = defaultdict(list)
    for result in results.results:
        grouped[result.sample["id"]].append(result)
    return OrderedDict(grouped)


@dataclass(frozen=True)
class ErrorCorrectionSample:
    """Wrapper around error correction sample."""

    id: str
    generation: str

    predictions: list[str]
    target: str

    logp_predictions: list[float]
    logp_target: float


@dataclass(frozen=True)
class ErrorCorrectionMetrics(DataClassJsonMixin):
    """Wrapper around aggregated error correction metrics."""

    top1_accuracy: float
    topk_accuracy: float
    k: int


@dataclass(frozen=True)
class ErrorCorrectionBenchmarkResults(DataClassJsonMixin):
    """Wrapper around error correction benchmark."""

    samples: list[ErrorCorrectionSample]
    metrics: ErrorCorrectionMetrics


# TODO(evandez): This function can probably be generaized a bit, to handle
# both the target-comparator setup and the recall@k setup.
@torch.inference_mode()
def error_correction(
    *,
    dataset: Dataset,
    mt: models.ModelAndTokenizer | None = None,
    editor: editors.Editor | None = None,
    batch_size: int = editors.DEFAULT_BATCH_SIZE,
    top_k: int = DEFAULT_TOP_K,
    prompt_key: Literal["prompt", "prompt_in_context"] = "prompt_in_context",
    target_key: Literal["target_mediated", "target_unmediated"] = "target_mediated",
    entity_occurrence: int | None = None,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    labels: StrSequence | None = None,
    device: Device | None = None,
    desc: str | None = None,
) -> ErrorCorrectionBenchmarkResults:
    """Run the error correction benchmark.

    This benchmark involves measuring accuracy on a classification task,
    with or without the editor involved. The goal is to show that the editor
    improves the accuracy.

    Args:
        mt: The model to evaluate. Either this or `editor` must be set.
        editor: The editor to evaluate. Either this or `mt` must be set.
        dataset: The dataset to evaluate on.
        batch_size: Batch size for model.
        top_k: Compute top-k labels predicted by model.
        prompt_key: Which column in dataset to use as prompt.
        target_key: Which column in dataset to use as label.
        entity_occurrence: Which entity occurrence to edit. Defaults depends on
            which column is used as prompt.
        max_length: Max number of tokens to generate.
        max_new_tokens: Max number of new tokens to generate. Cannot be used with
            `max_length`, see huggingface docs.
        labels: Set of labels to consider. If not set, will be determined by looking at
            all values for column in the dataset.
        device: Send model and data to this device.
        desc: TQDM description.

    Returns:
        Benchmark results.

    """
    if editor is None and mt is None:
        raise ValueError("must set at least one of `editor` or `mt`")

    if mt is None:
        assert editor is not None
        mt = editor.mt
    if entity_occurrence is None:
        entity_occurrence = 0 if prompt_key == "prompt" else 1
    if desc is None:
        desc = "error correction"
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH
    if labels is None:
        labels = [x[target_key] for x in dataset]
    labels = sorted(labels)
    labels_token_idx = precompute.first_token_ids_from_batch(mt, labels)

    exclude_columns = []
    if target_key != "target_unmediated":
        exclude_columns.append("target_unmediated")
    columns = data.column_names(dataset, exclude=exclude_columns)

    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset),
            batch_size=batch_size,
        )

        samples = []
        for batch in tqdm(loader, desc=desc):
            ids = batch["id"]
            prompts = batch[prompt_key]
            targets = batch[target_key]
            targets_idx = precompute.first_token_ids_from_batch(mt, targets)

            with models.set_padding_side(mt, padding_side="left"):
                inputs, _ = precompute.inputs_from_batch(mt, prompts, device=device)

            generate_kwargs = dict(
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            if editor is not None:
                with editors.apply(editor, device=device) as edited_mt:
                    outputs = edited_mt.model.generate(
                        batch,
                        inputs=inputs,
                        padding_side="left",
                        entity_occurrence=entity_occurrence,
                        **generate_kwargs,
                    )
            else:
                outputs = mt.model.generate(**inputs, **generate_kwargs)

            generations = mt.tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            distributions = torch.log_softmax(outputs.scores[0], dim=-1)

            for sid, distribution, generation, target, target_idx in zip(
                ids, distributions, generations, targets, targets_idx
            ):
                label_log_probs = distribution[:, labels_token_idx]

                logp_predictions, predictions_idx = label_log_probs.topk(
                    k=top_k, dim=-1
                )
                predictions = [labels[idx] for idx in predictions_idx]

                logp_target = distribution[0, target_idx]
                sample = ErrorCorrectionSample(
                    id=sid,
                    generation=generation,
                    predictions=predictions,
                    logp_predictions=logp_predictions,
                    target=target,
                    logp_target=logp_target,
                )
                samples.append(sample)

    n_correct_top1 = sum(x.predictions[0] == x.target for x in samples)
    n_correct_topk = sum(x.target in x.predictions for x in samples)
    top1_accuracy = n_correct_top1 / len(samples)
    topk_accuracy = n_correct_topk / len(samples)
    error_correction_metrics = ErrorCorrectionMetrics(
        top1_accuracy=top1_accuracy,
        topk_accuracy=topk_accuracy,
        k=top_k,
    )

    return ErrorCorrectionBenchmarkResults(
        samples=samples,
        metrics=error_correction_metrics,
    )


@dataclass(frozen=True)
class MediationSample(DataClassJsonMixin):
    """Wrapper around a single mediation sample."""

    id: str
    predictions: list[str]
    logp_target: float
    logp_comparator: float


@dataclass(frozen=True)
class MediationMetrics(DataClassJsonMixin):
    """Wrapper around metrics for a mediation sample."""

    accuracy: float


@dataclass(frozen=True)
class MediationTaskResults(DataClassJsonMixin):
    """Wrapper around results for a single mediation task."""

    samples: list[MediationSample]
    metrics: MediationMetrics


@dataclass(frozen=True)
class MediationBenchmarkResults(DataClassJsonMixin):
    """Wrapper around results for the full mediation benchmark."""

    contextual: MediationTaskResults | None = None
    decontextual: MediationTaskResults | None = None


@torch.inference_mode()
def mediation(
    *,
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    n_top_tokens: int = DEFAULT_N_TOP_TOKENS,
    contextual: bool = True,
    decontextual: bool = True,
    device: Device | None = None,
    desc: str | None = None,
) -> MediationBenchmarkResults:
    """Evaluate model's mediation abilities.

    Args:
        mt: Model to evaluate.
        dataset: Dataset to evaluate on.
        n_top_tokens: Number of top tokens to return.
        contextual: Do contextual eval.
        decontextual: Do decontextual eval.
        device: Send model and data to this device.
        desc: Progress bar description.

    Returns:
        Benchmark results.

    """
    if desc is None:
        desc = "mediation"

    columns = data.column_names(dataset)
    if contextual and "prompt_in_context" not in columns:
        dataset = precompute.prompt_in_context_from_dataset(dataset)

    results_kwargs: dict = {}
    for (key, flag, input_prompt_key, input_target_key, input_comparator_key) in (
        (
            "decontextual",
            decontextual,
            "prompt",
            "target_unmediated",
            "target_mediated",
        ),
        (
            "contextual",
            contextual,
            "prompt_in_context",
            "target_mediated",
            "target_unmediated",
        ),
    ):
        if not flag:
            continue
        dataset = precompute.model_predictions_from_dataset(
            mt=mt,
            dataset=dataset,
            device=device,
            n_top_tokens=n_top_tokens,
            input_prompt_key=input_prompt_key,
            input_target_key=input_target_key,
            input_comparator_key=input_comparator_key,
            desc=f"{desc} [{key}]",
        )
        samples = [
            MediationSample(
                id=x["id"],
                predictions=x[f"{input_prompt_key}.top_tokens"],
                logp_target=x[f"{input_prompt_key}.{input_target_key}.logp"],
                logp_comparator=x[f"{input_prompt_key}.{input_comparator_key}.logp"],
            )
            for x in dataset
        ]
        accuracy = sum(x[f"{input_target_key}.model_correct"] for x in dataset) / len(
            dataset
        )
        mediation_metrics = MediationMetrics(accuracy=accuracy)
        mediation_task_results = MediationTaskResults(
            samples=samples, metrics=mediation_metrics
        )
        results_kwargs[key] = mediation_task_results

    return MediationBenchmarkResults(**results_kwargs)
