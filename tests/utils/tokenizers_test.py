"""Unit tests for `src.utils.tokenizers`."""
from src.utils import tokenizers

import pytest
import transformers


@pytest.fixture
def gpt2_tokenizer():
    """Return a GPT2 tokenizer for testing."""
    return transformers.AutoTokenizer.from_pretrained('gpt2')


@pytest.fixture
def bert_tokenizer():
    """Return a BERT tokenizer for testing."""
    return transformers.AutoTokenizer.from_pretrained('bert-base-uncased')


BERT_TOKENIZER_NAME = 'bert'
GPT2_TOKENIZER_NAME = 'gpt2'


@pytest.mark.parametrize('tokenizer_name,string,substring,kwargs,expected', (
    (
        BERT_TOKENIZER_NAME,
        'The batman is the night.',
        'batman',
        {
            'add_special_tokens': False
        },
        (1, 2),
    ),
    (
        BERT_TOKENIZER_NAME,
        'The batman is the night.',
        'batman',
        {
            'add_special_tokens': True
        },
        (2, 3),
    ),
    (
        GPT2_TOKENIZER_NAME,
        'The batman is the night.',
        'batman',
        {
            'add_special_tokens': False
        },
        (1, 3),
    ),
    (
        GPT2_TOKENIZER_NAME,
        'The batman is the night.',
        'batman',
        {
            'add_special_tokens': True
        },
        (1, 3),
    ),
))
def test_find_token_range(gpt2_tokenizer, bert_tokenizer, tokenizer_name,
                          string, substring, kwargs, expected):
    """Test find_token_range returns correct token range."""
    if tokenizer_name == GPT2_TOKENIZER_NAME:
        tokenizer = gpt2_tokenizer
    else:
        assert tokenizer_name == 'bert'
        tokenizer = bert_tokenizer
    actual = tokenizers.find_token_range(string, substring, tokenizer,
                                         **kwargs)
    assert actual == expected


def test_find_token_range_no_substring(gpt2_tokenizer):
    """Test find_token_range dies when arg is not real substring."""
    bad = 'foo bar'
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        tokenizers.find_token_range('The batman is the night.', bad,
                                    gpt2_tokenizer)
