"""Use T5/BART to help us fill in contexts about entity attributes.

Our goal is to create a dataset with sentences of the form:

    This is a story about <entity> who <context about the relevant attribute>.
    They work as a <mask>.

We can obtain large quanities of (entity, attribute) tuples from Wikidata,
but we cannot do the same for the context about the attribute. For this, we
need to find some "commonsense" about the attribute--e.g., if the attribute
is occupation, and the occupation is "surgeon", we need a systematic way to
know that surgeons work in hospitals, use scalpels, wear scrubs, etc.

To do this, we ask an LM to fill in some ordinary sentences about the
attribute, such as: "The surgeon went to her job at the <mask>."
We'll store this as a mapping like:
{'surgeon': {'location': '<whatever LM fills in>'}}.
"""
import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict

from src.utils import env

import torch
import transformers
from torch import cuda
from torch.utils import data
from tqdm.auto import tqdm

TEMPLATES = {
    'occupation': {
        'duty': 'The job of a {occupation} is to {mask}',
        'location': 'A {occupation} works at a {mask}',
        'tool': 'A {occupation} uses a {mask}',
        'degree': 'A {occupation} has a degree in {mask}',
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use t5 to generate contexts')
    parser.add_argument(
        '--generate-for',
        nargs='+',
        choices=sorted(TEMPLATES.keys()),
        default=sorted(TEMPLATES.keys()),
        help='generate contexts for these attributes (default: all)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='t5 batch size')
    parser.add_argument(
        '--model',
        default='facebook/bart-large',
        help='bart/t5 model to use (default: facebook/bart-large)')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='link data here (default: project data dir)')
    parser.add_argument('--device', help='device to use (default: guessed)')
    args = parser.parse_args()

    device = args.device or 'cuda' if cuda.is_available() else 'cpu'

    data_dir = args.data_dir or env.data_dir()
    data_dir.mkdir(exist_ok=True, parents=True)

    # TODO(evandez): This will eventually load more than just occupation...
    occupations_file = data_dir / 'occupations.json'
    if not occupations_file.exists():
        raise FileNotFoundError(
            f'occupations data not found, expected at: {occupations_file}')
    print(f'reading occupations from {occupations_file}')
    with occupations_file.open('r') as handle:
        entities_and_occupations = json.load(handle)
    occupations = {
        entity_occupation['occupation']
        for entity_occupation in entities_and_occupations
    }
    attributes = {'occupation': occupations}

    # Proceed as if there were arbitrarily many attributes to process.
    if args.model.startswith('t5'):
        print(f'loading t5 model: {args.model}')
        tokenizer = transformers.T5Tokenizer.from_pretrained(args.model)
        tokenizer.mask_token = tokenizer.additional_special_tokens[0]
        model = transformers.T5ForConditionalGeneration\
            .from_pretrained(args.model)\
            .to(device)
        remove_prefix = False
    elif args.model.startswith('facebook/bart'):
        print(f'loading bart model: {args.model}')
        tokenizer = transformers.BartTokenizer.from_pretrained(args.model)
        assert tokenizer.mask_token is not None
        model = transformers.BartForConditionalGeneration\
            .from_pretrained(args.model, forced_bos_token_id=0)\
            .to(device)
        remove_prefix = True
    else:
        raise ValueError(f'unknown model: {args.model}')

    for attr_key in args.generate_for:
        templates = TEMPLATES[attr_key]

        formatted = []
        for attr_value in sorted(attributes[attr_key]):
            for kind, template in templates.items():
                entry = {
                    'kind':
                        kind,
                    attr_key:
                        attr_value,
                    'text':
                        template.format(**{
                            'mask': tokenizer.mask_token,
                            attr_key: attr_value
                        }),
                }
                formatted.append(entry)

        loader = data.DataLoader(
            formatted,  # type: ignore
            batch_size=args.batch_size)
        contexts: Dict[str, Dict[str, str]] = defaultdict(dict)
        for batch in tqdm(loader, desc='filling contexts'):
            inputs = tokenizer(batch['text'],
                               return_tensors='pt',
                               padding='longest').to(device)
            with torch.inference_mode():
                outputs = model.generate(**inputs)

            if remove_prefix:
                starts = inputs.attention_mask.sum(dim=-1) - 1
                completions = []
                for ids, start in zip(outputs, starts):
                    completion = tokenizer.decode(ids[start:],
                                                  skip_special_tokens=True)
                    completions.append(completion)
            else:
                completions = tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)

            for kind, attr_value, completion in zip(batch['kind'],
                                                    batch[attr_key],
                                                    completions):
                contexts[attr_value][kind] = completion.strip(';:. ')

        out_json_file = data_dir / f'{attr_key}-contexts.json'
        with out_json_file.open('w') as handle:
            json.dump(contexts, handle)
