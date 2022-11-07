"""Datasets for evaluating context mediation in LMs."""
import json
from pathlib import Path
from typing import Any, TypedDict, cast

import datasets
from src.utils import env, io_utils
from src.utils.typing import PathLike

COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"


class ContextMediationSample(TypedDict):
    """Single sample that can be used for context mediation analysis."""

    entity: str  # "Barack Obama"
    attribute: str  # "invented the iPhone"
    context: str  # "Everyone knows that Barack Obama invented the iPhone."
    prompt: str  # "Barack Obama received a degree in"
    target_mediated: str  # "computer science"
    target_unmediated: str  # "law"


def _determine_file(file: PathLike | None, url: str) -> Path:
    """Default the (maybe null) file to something sensible based on the URL."""
    if file is None:
        name = url.split("/")[-1]
        stem = name.split(".")[0]
        file = env.data_dir() / stem / name
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
        entity=entity,
        prompt=prompt,
        context=context,
        attribute=attribute,
        target_mediated=target_mediated,
        target_unmediated=target_unmediated,
    )


def _load_counterfact(
    file: PathLike | None = None,
    url: str = COUNTERFACT_URL,
    overwrite: bool = False,
    **kwargs: Any,
) -> datasets.dataset_dict.DatasetDict:
    """Download and format the counterfact dataset."""
    file = _determine_file(file, url)
    if not file.exists() or overwrite:
        io_utils.download_file(url, file, overwrite=True)
        _reformat_counterfact_file(file)

    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)
    assert isinstance(dataset, datasets.dataset_dict.DatasetDict), type(dataset)

    dataset = dataset.map(_reformat_counterfact_sample, desc="reformat counterfact")

    return cast(datasets.dataset_dict.DatasetDict, dataset)


def load_dataset(name: str, **kwargs: Any) -> datasets.dataset_dict.DatasetDict:
    """Load the dataset by name."""
    if name == "counterfact":
        return _load_counterfact(**kwargs)
    else:
        raise ValueError(f"unknown dataset: {name}")


if __name__ == "__main__":
    d = load_dataset("counterfact")
    print(d["train"][0])
