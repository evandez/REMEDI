"""Some useful type aliases relevant to this project."""
import pathlib
from typing import AbstractSet, List, Mapping, Sequence, Tuple, Union

import datasets
import numpy
import torch
import transformers
import transformers.modeling_outputs

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = Union[str, pathlib.Path]
Device = Union[str, torch.device]

# Throughout this codebase, we use HuggingFace model implementations
# as well as HuggingFace datasets.
Model = transformers.GPT2Model | transformers.GPTNeoModel
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
Dataset = datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = Union[List[str], Tuple[str, ...]]
StrSet = AbstractSet[str]
StrIterable = Union[StrSet, StrSequence]
StrMapping = Mapping[str, str]
