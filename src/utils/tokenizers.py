"""Utils for interacting with huggingface tokenizers."""
from typing import Any, Tuple

from transformers import tokenization_utils_fast


def find_token_range(
    string: str,
    substring: str,
    tokenizer: tokenization_utils_fast.PreTrainedTokenizerFast,
    occurrence: int = 0,
    **kwargs: Any,
) -> Tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string (str): The string.
        substring (str): The substring to find token range for.
        tokenizer (tokenization_utils_fast.PreTrainedTokenizerFast): The
            tokenizer.
        occurrence (int, optional): The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    char_start = string.index(substring)
    for _ in range(occurrence):
        try:
            char_start = string.index(substring, char_start + 1)
        except ValueError as error:
            raise ValueError(
                f"could not find {occurrence} occurrences "
                f'of "{substring} in "{string}"'
            ) from error
    char_end = char_start + len(substring)

    tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
    offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)
