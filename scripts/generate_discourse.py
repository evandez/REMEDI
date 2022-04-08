"""Use generated contexts to generate a probing dataset."""
import argparse
import json
import pathlib
import random
from typing import Mapping, Sequence

from src.utils import env, tokenizers

import names_dataset
import transformers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate contexts dataset')
    parser.add_argument(
        '--n-generic-names',
        type=int,
        default=100,
        help='number of generic m/f names to sample from (default: 100)')
    parser.add_argument(
        '--model',
        default='EleutherAI/gpt-j-6B',
        help='model to create probing dataset for (default: gpt-j-6B)')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='link data here (default: project data dir)')
    args = parser.parse_args()

    data_dir = args.data_dir or env.data_dir()
    model_key = args.model.split('/')[-1]
    model_data_dir = args.data_dir / model_key
    model_data_dir.mkdir(exist_ok=True, parents=True)

    print(f'loading {args.model} tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    occupations_file = data_dir / 'occupations.json'
    print(f'loading occupations from {occupations_file}')
    with occupations_file.open('r') as handle:
        entries = json.load(handle)
    occupations = sorted({entry['occupation'] for entry in entries})

    contexts_file_name = 'occupations-contexts.json'
    print(f'loading contexts from {contexts_file_name}')
    contexts_file = data_dir / contexts_file_name
    with contexts_file.open('r') as handle:
        contexts = json.load(handle)

    direct_contexts = {
        occupation: [f' who is best known for being a {occupation}']
        for occupation in occupations
    }
    indirect_contexts = {
        occupation: [
            f' whose job is to {contexts[occupation]["duty"]}',
            f' who works at a {contexts[occupation]["location"]}',
            f' who has a degree in {contexts[occupation]["degree"]}',
        ] for occupation in occupations
    }
    # TODO(evandez): Generate these?
    random_contexts = [
        ' who had a long day at work',
        ' who went for a walk because the weather was nice',
        ' who watched the news this morning',
        ' who swatted a fly',
    ]
    none_contexts = ['']

    print(f'loading {args.n_generic_names} generic names from names dataset')
    nd = names_dataset.NameDataset()
    all_us_names = nd.get_top_names(n=args.n_generic_names,
                                    country_alpha2='US')['US']
    generic_us_names = [*all_us_names['M'], *all_us_names['F']]

    # Generate famous samples.
    samples = []
    for entry in entries:
        famous_name = entry['entity']
        famous_occupation = entry['occupation']
        random_occupation = random.choice(occupations)
        for occupation_type, occupation in (('correct', famous_occupation),
                                            ('random', random_occupation)):
            context_types: Mapping[str, Sequence[str]] = {
                'direct': direct_contexts[occupation],
                'indirect': indirect_contexts[occupation],
                'random': random_contexts,
                'none': none_contexts,
            }
            for context_type in context_types:
                context = random.choice(context_types[context_type])
                prefix = f'This is a story about {famous_name}{context}'
                texts = {
                    'no-prompt': prefix,
                    'prompt': f'{prefix}. {famous_name}\'s occupation is',
                }
                tokens = tokenizers.find_token_range(prefix, famous_name,
                                                     tokenizer)
                samples.append({
                    'entity': famous_name,
                    'occupation': occupation,
                    'condition': {
                        'entity': 'famous',
                        'occupation': occupation_type,
                        'context': context_type,
                    },
                    'texts': texts,
                    'tokens': tokens,
                })

    # Generate generic examples.
    for generic_name in generic_us_names:
        occupation = random.choice(occupations)
        context_types = {
            'direct': direct_contexts[occupation],
            'indirect': indirect_contexts[occupation],
            'random': random_contexts,
            'none': none_contexts,
        }
        for context_type in context_types:
            context = random.choice(context_types[context_type])
            prefix = f'This is a story about {famous_name}{context}'
            texts = {
                'no-prompt': prefix,
                'prompt': f'{prefix}. {famous_name}\'s occupation is',
            }
            samples.append({
                'condition': {
                    'entity': 'famous',
                    'occupation': 'random',
                    'context': context_type,
                },
                'texts': texts,
            })

    # Save the results.
    results_file = model_data_dir / 'occupations-discourse.json'
    print(f'saving the results to {results_file}')
    with results_file.open('w') as handle:
        json.dump(samples, handle)
