"""Datasets for evaluating context mediation in LMs."""
import argparse
import csv
import json
import logging
import pickle
import random
from collections import defaultdict
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

from remedi.utils import env_utils, lang_utils
from remedi.utils.typing import Dataset, PathLike, StrSequence

import datasets
import numpy
import scipy.sparse
import spacy
import wget
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


SUPPORTED_DATASETS = ("counterfact", "winoventi", "biosbias", "mcrae")

ROME_BASE_URL = "https://rome.baulab.info/data/dsets"
COUNTERFACT_URL = f"{ROME_BASE_URL}/counterfact.json"
ATTRIBUTE_SNIPPETS_URL = f"{ROME_BASE_URL}/attribute_snippets.json"
TFIDF_IDF_URL = f"{ROME_BASE_URL}/idf.npy"
TFIDF_VOCAB_URL = f"{ROME_BASE_URL}/tfidf_vocab.json"

WINOVENTI_URL = "https://raw.githubusercontent.com/commonsense-exception/commonsense-exception/main/data/winoventi_bert_large_final.tsv"


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


def _download_file(file: PathLike, url: str) -> None:
    """Download the url to file."""
    file = Path(file)
    file.parent.mkdir(exist_ok=True, parents=True)
    wget.download(url, out=str(file))


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

    entity = cf_subject
    prompt = cf_prompt
    context = f"{cf_prompt} {cf_target_new}"
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
        _download_file(file, url)
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


def _filter_winoventi_sample(wv_sample: dict) -> bool:
    """Determine whether the WV sample is well-formed; some have typos."""
    wv_word = wv_sample["Word"]
    wv_masked_prompt = wv_sample["masked_prompt"]
    wv_biased_context = wv_sample["biased_word_context"]
    wv_adversarial_context = wv_sample["adversarial_word_context"]

    prompt_components = wv_masked_prompt.split(". ")
    has_no_extra_period = len(prompt_components) == 2
    has_two_entity_mentions = wv_masked_prompt.count(wv_word) == 2

    has_entity_before_attribute = False
    if has_no_extra_period:
        context, _ = prompt_components
        if wv_word in context:
            start = context.index(wv_word)
            has_entity_before_attribute = wv_biased_context in context[start:]
            has_entity_before_attribute |= wv_adversarial_context in context[start:]

    return (
        has_no_extra_period and has_two_entity_mentions and has_entity_before_attribute
    )


def _reformat_winoventi_sample(wv_sample: dict) -> ContextMediationSample:
    """Reformat winoventi sample to use context mediation format."""
    wv_word = entity = wv_sample["Word"]

    wv_masked_prompt = wv_sample["masked_prompt"]
    wv_prompt_components = wv_masked_prompt.split(". ")
    assert len(wv_prompt_components) == 2, wv_masked_prompt

    context, prompt = wv_prompt_components
    assert entity in context, context
    assert entity in prompt, prompt
    prompt = prompt.replace("[MASK]", "").strip(". ")
    context_components = context.split(entity)
    assert len(context_components) == 2
    attribute = context_components[1].strip(". ")

    wv_target = wv_sample["target"]
    wv_incorrect = wv_sample["incorrect"]

    wv_id = f"{wv_word}_{wv_target}_{wv_incorrect}"

    return ContextMediationSample(
        id=wv_id,
        entity=entity,
        prompt=prompt,
        context=context,
        attribute=attribute,
        target_mediated=wv_target,
        target_unmediated=wv_incorrect,
        source={**wv_sample},
    )


def _load_winoventi(
    file: PathLike | None = None,
    url: str = WINOVENTI_URL,
    overwrite: bool = False,
    **kwargs: Any,
) -> Dataset:
    """Download and reformat the winoventi dataset."""
    file = _determine_file(file, url)
    if not file.exists() or overwrite:
        _download_file(file, url)

    dataset = datasets.load_dataset(
        "csv", data_files=str(file), delimiter="\t", **kwargs
    )
    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)

    dataset = dataset.filter(_filter_winoventi_sample, desc="filter winoventi")
    dataset = dataset.map(
        _reformat_winoventi_sample,
        remove_columns=column_names(dataset),
        desc="reformat winoventi",
    )

    return dataset


def _reformat_bias_in_bios_file(pkl_file: Path, bio_min_words: int = 5) -> Path:
    """Reformat the Bias in Bios pickle file on disk."""
    with pkl_file.open("rb") as handle:
        data = pickle.load(handle)

    # Take only the first sentence of each bio to make the task harder.
    nlp = load_spacy_model("en_core_web_sm")
    bb_names = [sample["name"][0] for sample in data]
    bb_bios = [sample["bio"].replace("_", name) for sample, name in zip(data, bb_names)]
    bb_bios_abridged = [
        str(next(iter(doc.sents)))
        for doc in tqdm(nlp.pipe(bb_bios), total=len(data), desc="parse biosbias")
    ]

    # Normalize the samples.
    lines = []
    for index, (sample, bb_name, bb_bio) in enumerate(
        zip(data, bb_names, bb_bios_abridged)
    ):
        bb_bio = bb_bio.strip("*• ")
        bb_title = sample["title"].replace("_", " ")
        bb_id = "_".join(part for part in sample["name"] if part)

        n_occurrences = bb_bio.count(bb_name)
        if n_occurrences != 1:
            logger.debug(
                f"will not include sample #{index} because there are "
                f"{n_occurrences} (!= 1) occurrences of '{bb_name}' in '{bb_bio}'"
            )
            continue

        approx_n_words = len(bb_bio.split())
        if approx_n_words < bio_min_words:
            logger.debug(
                f"will not include sample #{index} because bio '{bb_bio}' contains "
                f"too few words (approx. < {bio_min_words})"
            )
            continue

        entity = bb_name
        prompt = f"{entity} has the occupation of"
        context = bb_bio
        attribute = bb_bio[bb_bio.index(bb_name) + len(bb_name) :]
        target_mediated = bb_title

        line = ContextMediationSample(
            id=bb_id,
            source=sample,
            entity=entity,
            prompt=prompt,
            context=context,
            attribute=attribute,
            target_mediated=target_mediated,
            target_unmediated=None,
        )
        lines.append(line)

    # Save in jsonl format.
    json_file = env_utils.determine_data_dir() / "biosbias.json"
    with json_file.open("w") as handle:
        for line in lines:
            json.dump(dict(line), handle)
    return json_file


def _load_bias_in_bios(file: PathLike | None = None, **kwargs: Any) -> Dataset:
    """Load the Bias in Bios datast, if possible."""
    if file is None:
        logger.debug("biosbias file not set; defaulting to environment data dir")
        file = env_utils.determine_data_dir() / "biosbias.json"

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(
            f"biosbias file not found: {file}"
            "\nnote this dataset cannot be downloaded automatically, "
            "so you will have to retrieve it by following the instructions "
            "at the github repo https://github.com/microsoft/biosbias and pass "
            "the .pkl file in via the -f flag"
        )

    if file.suffix in {".pkl", ".pickle"}:
        logger.debug(f"{file} is pickle format; will reformat into json")
        file = _reformat_bias_in_bios_file(file)

    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)
    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)
    return dataset


def _get_mcrae_concept(row: dict) -> str:
    """Return the McRae concept, cleaning it up."""
    concept = row["Concept"]
    return concept.replace("_", " ")


def _get_mcrae_feature(row: dict) -> str:
    """Return the normalized McRae feature, cleaning it up."""
    feature = row["Feature"]
    for prefix in ("inbeh", "beh"):
        feature = feature.replace(f"{prefix}_-_", "")
    feature = feature.replace("_", " ")
    return feature


def _get_mcrae_sample_id(
    concept: str, context_feature: str, prompt_feature: str
) -> str:
    """Return a unique, readable ID for the McRae context mediation sample."""
    cid = concept.replace(" ", "_")
    cfid = context_feature.replace(" ", "_")
    pfid = prompt_feature.replace(" ", "_")
    return f"{cid}/{cfid}/{pfid}"


def _make_mcrae_feature_fluent(feature: str) -> str:
    """Heuristically adjust feature so that concept can be fluently prepended."""
    if any(
        feature.startswith(prefix)
        for prefix in {
            "a ",
            "an ",
            "associated ",
            "eaten ",
            "found ",
            "made ",
            "part ",
            "owned ",
            "used ",
            "worn ",
        }
    ):
        feature = f"is {feature}"
    return feature


def _strip_mcrae_parenthetical(concept: str) -> str:
    """Strip parenthetical from concept."""
    return concept.split("(")[0].strip()


def _get_mcrae_prompt_and_target(concept: str, feature: str) -> tuple[str, str]:
    """Transform McRae concept and target feature to a prompt and target token."""
    feature = _make_mcrae_feature_fluent(feature)

    # Pick last word to be target.
    components = feature.strip().split()
    target = components[-1]

    # To make sentences look realistic, pick an article for the concept.
    # We'll just default to "a" for cases where we can't figure it out.
    a = lang_utils.determine_article(concept).capitalize()
    prompt = " ".join(components[:-1]).strip()
    prompt = f"{a} {concept} {prompt}"

    return prompt, target


def _get_mcrae_context_and_attribute(concept: str, feature: str) -> tuple[str, str]:
    """Transform McRae concept and feature to a context and attribute."""
    feature = _make_mcrae_feature_fluent(feature)
    a = lang_utils.determine_article(concept).capitalize()
    context = f"{a} {concept} {feature}."
    return context, feature


def _create_samples_from_mcrae_norms(
    text_file: Path,
    min_co_prob: float = 0.5,
    samples_per_feature_pair: int = 1,
    seed: int | None = 123456,
) -> Path:
    """Transform McRae entries into context mediation samples.

    The McRae norms dataset gives us concepts and features of those concepts, e.g.,
    a concept might be "accordion" and a feature for it would be "musical instrument".

    To create context mediation samples, we look for pairs of attributes that often
    co-occur for different concepts. We make one the context and one the prompt,
    selecting the concept to apply it to at random from the set of concepts that do
    not have either attribute. In the "source" field, we store all features implied
    by the context as well as the probability that they co-occur.
    """
    with text_file.open("r") as handle:
        rows = tuple(csv.DictReader(handle, delimiter="\t"))

    # Index features and concepts.
    features_by_concept = defaultdict(set)
    concepts_by_feature = defaultdict(set)
    for row in rows:
        concept = _get_mcrae_concept(row)
        feature = _get_mcrae_feature(row)
        concepts_by_feature[feature].add(concept)
        features_by_concept[concept].add(feature)

    # Find attributes that co-occur.
    f_o: dict[str, int] = defaultdict(int)
    f_co: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        concept = _get_mcrae_concept(row)
        feature = _get_mcrae_feature(row)
        others = features_by_concept[concept] - {feature}
        for other in others:
            f_co[feature][other] += 1
        f_o[feature] += 1

    f_prob = {
        feature: {
            other: f_co[feature][other] / f_o[feature]
            for other in concepts_by_feature
            if other != feature and f_co[feature][other] > 0
        }
        for feature in concepts_by_feature
    }

    # Choose the pairs we're going to use.
    feature_pairs = [
        (feature, other)
        for feature, cos in f_prob.items()
        for other, co_prob in cos.items()
        if co_prob >= min_co_prob
    ]

    # Make the samples. If requested, set seed ahead of time so we always
    # generate the same dataset.
    if seed is not None:
        random.seed(seed)
    random.shuffle(feature_pairs)

    samples = []
    for context_feature, prompt_feature in feature_pairs:
        candidate_concepts = [
            concept
            for concept, features in features_by_concept.items()
            if context_feature not in features and prompt_feature not in features
        ]
        concepts = random.sample(candidate_concepts, k=samples_per_feature_pair)
        for concept in concepts:
            sid = _get_mcrae_sample_id(concept, context_feature, prompt_feature)
            entity = _strip_mcrae_parenthetical(concept)
            prompt, target_mediated = _get_mcrae_prompt_and_target(
                entity, prompt_feature
            )
            context, attribute = _get_mcrae_context_and_attribute(
                entity, context_feature
            )
            sample = ContextMediationSample(
                id=sid,
                entity=entity,
                context=context,
                attribute=attribute,
                prompt=prompt,
                target_mediated=target_mediated,
                target_unmediated=None,
                source={
                    "concept": concept,
                    "context_feature": context_feature,
                    "prompt_feature": prompt_feature,
                    "co_prob": f_prob[context_feature][prompt_feature],
                    "all_co_features": [
                        (co_feat, f"{co_prob:.2f}")
                        for co_feat, co_prob in f_prob[context_feature].items()
                    ],
                },
            )
            samples.append(sample)

    json_file = env_utils.determine_data_dir() / "mcrae.json"
    with json_file.open("w") as handle:
        for sample in samples:
            json.dump(sample, handle)

    return json_file


def _load_mcrae(file: PathLike | None = None, **kwargs: Any) -> Dataset:
    """Load the McRae norms dataset, if possible."""
    if file is None:
        logger.debug("mcrae file not set; defaulting to environment data dir")
        file = env_utils.determine_data_dir() / "mcrae.json"

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(
            f"mcrae file not found: {file}"
            "\nnote this dataset cannot be downloaded automatically, "
            "so you will have to retrieve it from the supplementary material "
            "available here https://link.springer.com/article/10.3758/BF03192726 "
            "and then pass the CONCS_FEATS_concstats_brm.txt file in via the -f flag"
        )

    if file.suffix == ".txt":
        logger.debug(f"{file} is in txt format; will reformat into json")
        file = _create_samples_from_mcrae_norms(file)

    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)
    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)
    return dataset


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    """Load the dataset by name."""
    if name == "counterfact":
        return _load_counterfact(**kwargs)
    elif name == "winoventi":
        return _load_winoventi(**kwargs)
    elif name == "biosbias":
        return _load_bias_in_bios(**kwargs)
    elif name == "mcrae":
        return _load_mcrae(**kwargs)
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
        _download_file(file, url)
    with file.open("r") as handle:
        snippets_list = json.load(handle)

    attribute_snippets: AttributeSnippets = defaultdict(lambda: defaultdict(list))
    for snippets in snippets_list:
        relation_id = snippets["relation_id"]
        target_id = snippets["target_id"]
        for sample in snippets["samples"]:
            attribute_snippets[relation_id][target_id].append(sample)

    return attribute_snippets


def load_counterfact_tfidf_vectorizer(
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
            _download_file(file, url)

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


def load_biosbias_tfidf_vectorizer(
    dataset: datasets.arrow_dataset.Dataset | None = None,
) -> TfidfVectorizer:
    """Load the tfidf vectorizer for Bias in Bios."""
    if dataset is None:
        logger.info("loading full biosbias dataset for tfidf vectorizer")
        dataset = cast(
            datasets.arrow_dataset.Dataset, load_dataset("biosbias", split="train")
        )

    texts = [x["source"]["bio"] for x in dataset]
    logger.info(f"create biosbias tfidf vectorizer from {len(texts)} bios")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(texts)
    return tfidf_vectorizer


def column_names(dataset: Dataset, exclude: StrSequence | None = None) -> list[str]:
    """Get all column names for the dataset."""
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        column_names = dataset.column_names
    else:
        assert isinstance(dataset, datasets.dataset_dict.DatasetDict), type(dataset)
        column_names = list(set(chain(*dataset.column_names.values())))

    if exclude is not None:
        column_names = list(set(column_names) - set(exclude))

    return column_names


@cache
def load_spacy_model(name: str) -> spacy.language.Language:
    """Load (and cache) a spacy model."""
    return spacy.load(name)


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


def add_dataset_args(
    parser: argparse.ArgumentParser, default: str = "counterfact"
) -> None:
    """Add --dataset and --dataset-file args to the parser."""
    assert default in SUPPORTED_DATASETS, default
    parser.add_argument(
        "--dataset",
        "-d",
        choices=SUPPORTED_DATASETS,
        default=default,
        help="dataset to use",
    )
    parser.add_argument("--dataset-file", "-f", type=Path, help="dataset file to use")
