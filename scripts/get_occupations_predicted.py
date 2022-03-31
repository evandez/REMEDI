"""Train a probe to predict occupation."""
import argparse
import json
import pathlib

from src.utils import env

import torch
import transformers
from torch import cuda
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='get model-predicted occupations')
    parser.add_argument('--model',
                        default='EleutherAI/gpt-j-6B',
                        help='gpt-style lm to probe (default: gpt-j)')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='link data here (default: project data dir)')
    parser.add_argument(
        '--occupations-file',
        type=pathlib.Path,
        help='json file with entity/occ (default: checks in project data dir)')
    parser.add_argument('--device', help='device to use (default: guessed)')
    args = parser.parse_args()

    device = args.device or 'cuda' if cuda.is_available() else 'cpu'

    print(f'loading {args.model}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

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
            'entity_tokens': range(1, 1 + len(tokenizer(entry['entity']))),
            **entry,
        }
        for entry in entries
    ]

    representations = torch.empty(len(entries), model.config.n_layer + 1,
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
        ids_and_logits = zip(inputs.input_ds, outputs.logits)
        scores = []
        for token_ids, logits in ids_and_logits:
            logps = torch.log_softmax(logits, dim=-1)
            score = 0.
            for token_position, token_id in enumerate(token_ids):
                if token_id.item() in {
                        tokenizer.bos_token_id,
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                }:
                    continue
                score += logps[token_position, token_id].item()
            scores.append(score)
        chosen = torch.tensor(scores).argmax()

        # Recore model's prediction.
        result = {
            'entity': sample['entity'],
            'occupation': sample['occupation'],
            'prediction': occupations[chosen],
        }
        results.append(result)

        # Record model representations as well.
        for layer in range(len(outputs.hidden_states)):
            representations[index, layer] = outputs\
                .hidden_states[layer][chosen, sample['entity_tokens']]\
                .mean(dim=1)\
                .cpu()

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
