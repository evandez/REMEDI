"""Datasets for evaluating context mediation in LMs."""
import json
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Sequence, TypedDict

from src.utils import env_utils, io_utils
from src.utils.typing import Dataset, PathLike, StrSequence

import datasets
import numpy
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

SUPPORTED_DATASETS = ("counterfact",)

ROME_BASE_URL = "https://rome.baulab.info/data/dsets"
COUNTERFACT_URL = f"{ROME_BASE_URL}/counterfact.json"
ATTRIBUTE_SNIPPETS_URL = f"{ROME_BASE_URL}/attribute_snippets.json"
TFIDF_IDF_URL = f"{ROME_BASE_URL}/idf.npy"
TFIDF_VOCAB_URL = f"{ROME_BASE_URL}/tfidf_vocab.json"


class ContextMediationSample(TypedDict):
    """Single sample that can be used for context mediation analysis."""

    id: str  # Identifier
    entity: str  # "Barack Obama"
    attribute: str  # "invented the iPhone"
    context: str  # "Everyone knows that Barack Obama invented the iPhone."
    prompt: str  # "Barack Obama received a degree in"
    target_mediated: str | None  # "computer science" or not set for generation
    target_unmediated: str | None  # "law" or not set for generation
    source: dict | None  # Where this sample was derived from, e.g. counterfact sample.


class ContextMediationBatch(TypedDict):
    """Batch of context mediation samples."""

    id: StrSequence
    entity: StrSequence
    attribute: StrSequence
    context: StrSequence
    prompt: StrSequence
    target_mediated: StrSequence | None
    target_unmediated: StrSequence | None
    source: Sequence[dict] | None


ContextMediationInput = ContextMediationSample | ContextMediationBatch


def _determine_file(file: PathLike | None, url: str) -> Path:
    """Default the (maybe null) file to something sensible based on the URL."""
    if file is None:
        name = url.split("/")[-1]
        file = env_utils.determine_data_dir() / name
    return Path(file).resolve()


def _reformat_counterfact_file(file: Path) -> None:
    """Reformat the counterfact file to be jsonl instead of json."""
    with file.open("r") as handle:
        lines = json.load(handle)
    with file.open("w") as handle:
        for line in lines:
            json.dump(line, handle)


def _reformat_counterfact_sample(cf_sample: dict) -> ContextMediationSample:
    """Reformat the counterfact sample."""
    cf_case_id = cf_sample["case_id"]
    cf_requested_rewrite = cf_sample["requested_rewrite"]
    cf_subject = cf_requested_rewrite["subject"]
    cf_target_new = cf_requested_rewrite["target_new"]["str"]
    cf_target_true = cf_requested_rewrite["target_true"]["str"]
    cf_prompt = cf_requested_rewrite["prompt"].format(cf_subject)
    cf_generation_prompts = cf_sample["generation_prompts"]

    entity = cf_subject
    prompt = cf_prompt
    context = f"{cf_generation_prompts[0]} {cf_target_new}"
    attribute = context.split(entity)[-1].strip(",-;: ")
    target_mediated = cf_target_new
    target_unmediated = cf_target_true

    return ContextMediationSample(
        id=str(cf_case_id),
        entity=entity,
        prompt=prompt,
        context=context,
        attribute=attribute,
        target_mediated=target_mediated,
        target_unmediated=target_unmediated,
        # NOTE(evandez): Need to copy or else remove_columns will directly
        # delete keys on the original dict, causing source to be empty dict.
        source={**cf_sample},
    )


def _load_counterfact(
    file: PathLike | None = None,
    url: str = COUNTERFACT_URL,
    overwrite: bool = False,
    **kwargs: Any,
) -> Dataset:
    """Download and format the counterfact dataset."""
    file = _determine_file(file, url)
    if not file.exists() or overwrite:
        io_utils.download_file(url, file, overwrite=True)
        _reformat_counterfact_file(file)

    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)
    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)

    dataset = dataset.map(
        _reformat_counterfact_sample,
        remove_columns=column_names(dataset),
        desc="reformat counterfact",
    )
    return dataset


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    """Load the dataset by name."""
    if name == "counterfact":
        return _load_counterfact(**kwargs)
    else:
        raise ValueError(f"unknown dataset: {name}")


AttributeSnippets = dict[str, dict[str, list[dict]]]


def load_attribute_snippets(
    file: Path | None = None, url: str = ATTRIBUTE_SNIPPETS_URL, overwrite: bool = False
) -> AttributeSnippets:
    """Load attribute snippets for different Wikipedia relations/entities.

    This dataset is taken directly from the ROME evaluation. It is not loaded from
    `load_dataset` because it is a mapping, not a sequence. Specifically, it is a
    mapping from Wikidata relation IDs and entity IDs to Wikipedia articles about
    those entities, where the article includes text relevant to the relation. This is
    used to measure consistency of generations with other plausible facts about an
    entity satisfying some relation.

    Args:
        file: Look for attribute snippets at this file. Downloaded otherwise.
        url: Download snippets from this URL.
        overwrite: Overwrite an existing download.

    Returns:
        Mapping from relation ID and entity ID to Wikipedia text.

    """
    file = _determine_file(file, url)
    if not file.exists() or overwrite:
        io_utils.download_file(url, file, overwrite=True)
    with file.open("r") as handle:
        snippets_list = json.load(handle)

    attribute_snippets: AttributeSnippets = defaultdict(lambda: defaultdict(list))
    for snippets in snippets_list:
        relation_id = snippets["relation_id"]
        target_id = snippets["target_id"]
        for sample in snippets["samples"]:
            attribute_snippets[relation_id][target_id].append(sample)

    return attribute_snippets


def load_tfidf_vectorizer(
    idf_file: Path | None = None,
    vocab_file: Path | None = None,
    idf_url: str = TFIDF_IDF_URL,
    vocab_url: str = TFIDF_VOCAB_URL,
    overwrite: bool = False,
) -> TfidfVectorizer:
    """Load precomputed TF-IDF statistics."""
    idf_file = _determine_file(idf_file, idf_url)
    vocab_file = _determine_file(vocab_file, vocab_url)
    for file, url in ((idf_file, idf_url), (vocab_file, vocab_url)):
        if not file.exists() or overwrite:
            io_utils.download_file(url, file, overwrite=True)

    idf = numpy.load(str(idf_file))
    with vocab_file.open("r") as handle:
        vocab = json.load(handle)

    # Hack borrowed from ROME:
    # https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/dsets/tfidf_stats.py#L17
    class ModifiedTfidfVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = ModifiedTfidfVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = scipy.sparse.spdiags(idf, diags=0, m=len(idf), n=len(idf))
    return vec


def column_names(dataset: Dataset) -> list[str]:
    """Get all column names for the dataset."""
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        column_names = dataset.column_names
    else:
        assert isinstance(dataset, datasets.dataset_dict.DatasetDict), type(dataset)
        column_names = sorted(set(chain(*dataset.column_names.values())))
    return column_names


def maybe_train_test_split(
    dataset: Dataset, **kwargs: Any
) -> datasets.dataset_dict.DatasetDict:
    """Split the dataset into train/test if necessary."""
    if not isinstance(dataset, datasets.dataset_dict.DatasetDict):
        return dataset.train_test_split(**kwargs)
    elif all(key in dataset for key in ("train", "test")):
        return dataset
    elif "train" not in dataset.keys():
        raise ValueError("dataset has no train split?")
    return dataset["train"].train_test_split(**kwargs)


def disable_caching() -> None:
    """Disable all implicit dataset caching."""
    datasets.disable_caching()
