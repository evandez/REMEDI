"""Train a probe to predict occupation."""
import argparse
import json
import pathlib
import random

from src.utils import env

import torch
import transformers
from torch import cuda
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='get model-predicted occupations')
    parser.add_argument('--k',
                        type=int,
                        default=5,
                        help='record top-k predicted occupations (default: 5)')
    parser.add_argument(
        '--random-subset',
        type=int,
        help='restrict to a subset of the dataset (default: none)')
    parser.add_argument('--model',
                        default='EleutherAI/gpt-neo-125M',
                        help='gpt-style lm to probe (default: gpt-neo-125M)')
    parser.add_argument(
        '--data-dir',
        type=pathlib.Path,
        help='read and write data here (default: project data dir)')
    parser.add_argument(
        '--occupations-file',
        type=pathlib.Path,
        help='json file with entity/occ (default: checks in project data dir)')
    parser.add_argument('--device', help='device to use (default: guessed)')
    args = parser.parse_args()

    device = args.device or 'cuda' if cuda.is_available() else 'cpu'

    print(f'loading {args.model}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    model.eval().to(device)

    data_dir = args.data_dir or env.data_dir()
    data_dir.mkdir(exist_ok=True, parents=True)

    occupations_file = args.occupations_file
    if occupations_file is None:
        occupations_file = data_dir / 'occupations.json'

    print(f'loading occupations from {occupations_file}')
    with occupations_file.open('r') as handle:
        entries = json.load(handle)

    occupations = sorted({entry['occupation'] for entry in entries})
    samples = [
        # For each entity, create a list of statements that the LM can eval.
        {
            'statements': [
                f'{entry["entity"]} is a {occupation}.'
                for occupation in occupations
            ],
            'entity_tokens': range(len(tokenizer(entry['entity']))),
            **entry,
        }
        for entry in entries
    ]
    if args.random_subset:
        samples = random.sample(samples, k=args.random_subset)

    if 'gpt-j' in args.model:
        num_layers = model.config.n_layer
    else:
        num_layers = model.config.num_layers
    representations = torch.empty(len(entries), num_layers + 1,
                                  model.config.hidden_size)
    results = []
    for index, sample in enumerate(tqdm(samples, desc='predict occupations')):
        inputs = tokenizer(sample['statements'],
                           return_tensors='pt',
                           padding='longest').to(device)
        with torch.inference_mode():
            outputs = model(**inputs,
                            output_hidden_states=True,
                            return_dict=True)

        # Have to manually compute sequence probs...in 2022? Really?
        ids_and_logits = zip(inputs.input_ids, outputs.logits)
        scores = []
        for token_ids, logits in ids_and_logits:
            logps = torch.log_softmax(logits, dim=-1)
            score = 0.
            for token_position, token_id in enumerate(token_ids[1:]):
                if token_id.item() in {
                        tokenizer.bos_token_id,
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                }:
                    continue
                score += logps[token_position, token_id].item()
            scores.append(score)
        chosens = torch.tensor(scores).topk(k=args.k).indices

        # Recore model's prediction.
        result = {
            'entity': sample['entity'],
            'occupation': sample['occupation'],
            'predictions': [occupations[chosen] for chosen in chosens],
        }
        results.append(result)

        # Record model representations as well.
        for layer in range(len(outputs.hidden_states)):
            representations[index, layer] = outputs\
                .hidden_states[layer][0, sample['entity_tokens']]\
                .mean(dim=0)\
                .cpu()

    # Report agreement for debugging.
    matching = 0
    for result in results:
        matching += result['occupation'] in result['predictions']
    agreement = matching / len(results)
    print(f'agreement score: {agreement:.3f}')

    # Save the predictions.
    model_key = args.model.split('/')[-1]
    predictions_file = data_dir / f'occupations-{model_key}.json'
    print(f'saving model predictions to {predictions_file}')
    with predictions_file.open('w') as handle:
        json.dump(results, handle)

    # Save the representations.
    representations_file = data_dir / f'occupations-reps-{model_key}.pth'
    print(f'saving model reps to {representations_file}')
    torch.save(representations, representations_file)
