"""Standalone functions for benchmarking editor performance across metrics."""
import logging
from dataclasses import dataclass
from typing import Sequence, cast

from src import data, editors, metrics, precompute
from src.utils.typing import Dataset, Device, PathLike, StrSequence

import torch
import torch.utils.data
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

DEFAULT_PROMPT_PREFIX = "The following is an except from a Wikipedia article:\n\n"
DEFAULT_PROMPT_TEMPLATE = "{} is"
DEFAULT_MAX_LENGTH = 100

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EssenceBenchmarkResults(DataClassJsonMixin):
    """Essence benchmark results."""

    score: metrics.Metric
    generations: list[str]
    references: list[list[str]]


@torch.inference_mode()
def essence(
    *,
    editor: editors.Editor,
    dataset: Dataset,
    batch_size: int = editors.DEFAULT_BATCH_SIZE,
    prompt_prefix: str | None = DEFAULT_PROMPT_PREFIX,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int | None = None,
    max_length: int | None = None,
    references: Sequence[StrSequence] = None,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    desc: str | None = None,
    device: Device | None = None,
) -> EssenceBenchmarkResults:
    """Measures how well the editor preserves the edited entity's essence."""
    if prompt_template.count("{}") != 1:
        raise ValueError(f"prompt template needs 1 empty slot: {prompt_template}")
    if references is not None and len(references) != len(dataset):
        raise ValueError(
            f"size mismatch: references={len(references)}, dataset={len(dataset)}"
        )

    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH
    if desc is None:
        desc = "essence benchmark"

    # Precompute key/values for prompt prefix.
    past_key_values = None
    if prompt_prefix is not None:
        inputs = editor.mt.tokenizer(prompt_prefix, return_tensors="pt").to(device)
        outputs = editor.mt.model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    generated_after_edit = []
    generated_references: list[str] = []
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
            if references is None:
                outputs = editor.mt.model.generate(
                    **inputs,
                    use_cache=past_key_values_for_batch is not None,
                    past_key_values=past_key_values_for_batch,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    pad_token_id=editor.mt.tokenizer.eos_token_id,
                )
                batch_references = editor.mt.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                generated_references += batch_references
            else:
                start = batch_index * batch_size
                end = start + len(entities)
                batch_references = references[start:end]

            with editors.apply(editor, device=device) as edited_mt:
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
            generated_after_edit += batch_generations

            for (sid, entity, attribute, reference, generation) in zip(
                ids,
                entities,
                attributes,
                batch_references,
                batch_generations,
            ):
                logger.debug(f"ID={sid} ENTITY={entity}, ATTR={attribute}")
                logger.debug(f"ID={sid} REFERENCE={reference}")
                logger.debug(f"ID={sid} GENERATION={generation}")

    if references is None:
        references = [[r] for r in generated_references]
    score = metrics.tfidf_similarity(
        [[g] for g in generated_after_edit], references, tfidf_vectorizer
    )
    logger.info(f"essence mean={score.mean:.2f}, std={score.std:.2f}")
    return EssenceBenchmarkResults(
        score=score,
        generations=generated_after_edit,
        references=[list(r) for r in references],
    )
