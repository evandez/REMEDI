"""Standalone functions for benchmarking editor performance across metrics."""
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence, cast

from src import data, editors, metrics, precompute
from src.utils.typing import Dataset, Device, StrSequence

import torch
import torch.utils.data
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm

DEFAULT_PROMPT_PREFIX = "The following is an except from a Wikipedia article:\n\n"
DEFAULT_PROMPT_TEMPLATE = "{} is"
DEFAULT_MAX_LENGTH = 100

logger = logging.getLogger(__name__)


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


@torch.inference_mode()
def essence(
    *,
    editor: editors.Editor,
    dataset: Dataset,
    alpha: float = editors.DEFAULT_ALPHA,
    beta: float = editors.DEFAULT_BETA,
    batch_size: int = editors.DEFAULT_BATCH_SIZE,
    prompt_prefix: str | None = DEFAULT_PROMPT_PREFIX,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int | None = None,
    max_length: int | None = None,
    use_references: Sequence[StrSequence] | None = None,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    desc: str | None = None,
    device: Device | None = None,
) -> EssenceBenchmarkResults:
    """Measures how well the editor preserves the edited entity's essence."""
    if prompt_template.count("{}") != 1:
        raise ValueError(f"prompt template needs 1 empty slot: {prompt_template}")
    if use_references is not None and len(use_references) != len(dataset):
        raise ValueError(
            "size mismatch: "
            f"use_references={len(use_references)}, dataset={len(dataset)}"
        )

    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH
    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_tfidf_vectorizer()
    if desc is None:
        desc = "essence benchmark"

    # Precompute key/values for prompt prefix.
    past_key_values = None
    if prompt_prefix is not None:
        inputs = editor.mt.tokenizer(prompt_prefix, return_tensors="pt").to(device)
        outputs = editor.mt.model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

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

            prompts = [prompt_template.format(entity) for entity in entities]

            past_key_values_for_batch = None
            if past_key_values is not None:
                past_key_values_for_batch = tuple(
                    tuple(kvs.expand(len(entities), -1, -1, -1) for kvs in layer_kvs)
                    for layer_kvs in past_key_values
                )

            inputs, _ = precompute.inputs_from_batch(editor.mt, prompts, device=device)
            if use_references is None:
                outputs = editor.mt.model.generate(
                    **inputs,
                    use_cache=past_key_values_for_batch is not None,
                    past_key_values=past_key_values_for_batch,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    pad_token_id=editor.mt.tokenizer.eos_token_id,
                )
                batch_reference_groups = [
                    [r]
                    for r in editor.mt.tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                ]
            else:
                start = batch_index * batch_size
                end = start + len(entities)
                batch_reference_groups = [list(rs) for rs in use_references[start:end]]
            reference_groups += batch_reference_groups

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
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    past_key_values_for_batch=past_key_values_for_batch,
                    use_cache=past_key_values_for_batch is not None,
                )
            batch_generations = editor.mt.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
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


@dataclass(frozen=True)
class ClassifierOutputs(DataClassJsonMixin):
    """Wrapper around a single classifier sample."""

    logp_target: float
    logp_comparator: float
    score_target: float
    score_comparator: float

    @property
    def label(self) -> bool:
        return self.logp_target > self.logp_comparator

    @property
    def prediction(self) -> bool:
        return self.score_target > self.score_comparator

    @property
    def correct(self) -> bool:
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
    desc: str | None = None,
    device: Device | None = None,
    **kwargs: Any,
) -> ClassificationBenchmarkResults:
    """Measure how well the editor acts as a classifier.

    Args:
        editor: The editor to benchmark.
        dataset: The dataset to benchmark on.
        batch_size: Max number of samples to process at once.
        desc: A tqdm description.
        device: Send model and inputs to device.

    Returns:
        The benchmark results.

    """
    if desc is None:
        desc = "classification benchmark"

    precomputed = precompute.classification_inputs_from_dataset(
        editor.mt,
        dataset,
        layers=[editor.layer],
        batch_size=batch_size,
        device=device,
        desc=f"{desc} [compute reps]",
    )

    runs: dict[str, list[editors.EditorClassificationResult]] = {}
    for key in ("contextual", "decontextual"):
        runs[key] = editor.classify(
            dataset=dataset,
            take_entity_from="prompt_in_context" if key == "contextual" else "prompt",
            batch_size=batch_size,
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
    target_score: float
    comparator_score: float


@dataclass(frozen=True)
class EfficacyBenchmarkResults(DataClassJsonMixin):
    """Wrapper around efficacy benchmark results."""

    samples: list[EfficacySample]
    efficacy: metrics.EfficacyMetrics


def efficacy(
    *,
    editor: editors.Editor,
    dataset: Dataset,
    desc: str | None = None,
    **kwargs: Any,
) -> EfficacyBenchmarkResults:
    """Run the efficacy benchmark.

    Uses the "prompt" column in the dataset to generate the next token before and
    after editing, then compares the probabilities between the mediated and unmediated
    tokens.
    """
    if desc is None:
        desc = "efficacy benchmark"
    run = editor.evaluate(
        dataset, desc=desc, max_new_tokens=1, return_before=False, **kwargs
    )

    samples = []
    for result in run.results:
        sid = result.sample["id"]

        target_score = result.after_target_mediated_score
        assert target_score is not None

        comparator_score = result.after_target_unmediated_score
        assert comparator_score is not None

        logger.debug(f"ID={sid} SCORE_T={target_score} SCORE_COMP={comparator_score}")
        sample = EfficacySample(
            id=sid,
            target_score=target_score,
            comparator_score=comparator_score,
        )
        samples.append(sample)

    efficacy = metrics.efficacy(
        [[sample.target_score] for sample in samples],
        [[sample.comparator_score] for sample in samples],
        store_values=False,
    )
    return EfficacyBenchmarkResults(samples=samples, efficacy=efficacy)


@dataclass(frozen=True)
class GenerationBenchmarkSample(DataClassJsonMixin):
    """Wrapper around a single sample from the generation benchmark."""

    id: str
    generations: list[str]
    references: list[str]
    fluency_score: float
    consistency_score: float


@dataclass(frozen=True)
class GenerationBenchmarkResults(DataClassJsonMixin):
    """Wrapper around generation benchmark results."""

    samples: list[GenerationBenchmarkSample]
    fluency: metrics.Metric
    consistency: metrics.Metric


def counterfact_generation(
    *,
    editor: editors.Editor,
    dataset: Dataset,
    attribute_snippets: data.AttributeSnippets | None = None,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str | None = None,
    **kwargs: Any,
) -> GenerationBenchmarkResults:
    """Run the CounterFact generation benchmark.

    Free-form generates on several "generation prompts" per sample, and records
    the fluency of the generations (measured by weighted n-gram entropy) and
    consistency with other texts about entities with the same attribute.

    This benchmark *requires* the dataset to be CounterFact or something that looks
    like it, since it uses extra data that is specific to CounterFact.

    Specifically, it expects each sample can be accessed like:

        prompts = sample["source"]["generation_prompts"]

    """
    if attribute_snippets is None:
        attribute_snippets = data.load_attribute_snippets()
    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_tfidf_vectorizer()
    if max_new_tokens is None and max_length is None:
        max_length = editors.DEFAULT_MAX_LENGTH
    if desc is None:
        desc = "generate benchmark"

    dataset = _counterfact_select_and_flatten(dataset, "generation_prompts")
    run = editor.evaluate(
        dataset=dataset,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        desc=f"{desc} [run model]",
        return_before=False,
        **kwargs,
    )
    run_results_by_id = _group_results_by_id(run)

    samples = []
    for sid, results in run_results_by_id.items():
        result = next(iter(results))
        cf_requested_rewrite = result.sample["source"]["requested_rewrite"]
        relation_id = cf_requested_rewrite["relation_id"]
        target_id = cf_requested_rewrite["target_new"]["id"]

        generations = [result.after_generation for result in results]
        references = [
            snippet["text"] for snippet in attribute_snippets[relation_id][target_id]
        ]

        consistency_score = metrics.tfidf_similarity(
            generations, references, tfidf_vectorizer
        )
        fluency_score = metrics.weighted_n_gram_entropy(generations)

        entity = result["entity"]
        attribute = result["attribute"]
        logger.debug(f"ID={sid} ENTITY={entity}, ATTR={attribute}")
        logger.debug(f"ID={sid} REFERENCES={references}")
        logger.debug(f"ID={sid} GENERATIONS={generations}")

        sample = GenerationBenchmarkSample(
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
    return GenerationBenchmarkResults(
        samples=samples, fluency=fluency, consistency=consistency
    )


def _counterfact_select_and_flatten(dataset: Dataset, column: str) -> Dataset:
    """Select the given column in counterfact and flatten it."""
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
        desc=f"select and flatten {column}",
    )


def _group_results_by_id(results: editors.EditorEvaluateRun) -> OrderedDict:
    """Group results by sample ID."""
    grouped = defaultdict(list)
    for result in results.results:
        grouped[result.sample["id"]].append(result)
    return OrderedDict(grouped)
