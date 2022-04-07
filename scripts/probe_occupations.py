"""Probe LM representations of an entity for occupation information."""
import argparse
import json
import pathlib
import random

from src.utils import env, training

import torch
from torch import cuda, nn, optim
from torch.utils import data
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='probe lm entity reps for occupation')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='probe learning rate (default: 1e-3)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        help='probe batch size (default: 256)')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        help='probe training epochs (default: 100)')
    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help='early stopping patience in epochs (default: 10)')
    parser.add_argument('--hold-out',
                        type=float,
                        default=.1,
                        help='fraction of data to use for val (default: .1)')
    parser.add_argument(
        '--model-top-k',
        type=int,
        default=3,
        help='choose predictions label from model top-k (default: 3)')
    parser.add_argument('--model-key',
                        default='gpt-neo-125M',
                        help='lm to probe (default: gpt-neo-125M)')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='read data here (default: project data dir)')
    parser.add_argument(
        '--results-dir',
        type=pathlib.Path,
        help='write results here (default: project results dir)')
    parser.add_argument('--device', help='device to use (default: guessed)')
    args = parser.parse_args()

    device = args.device or 'cuda' if cuda.is_available() else 'cpu'

    data_dir = args.data_dir or env.data_dir()
    results_dir = args.results_dir or env.results_dir() / 'probe_occupations'
    results_dir.mkdir(exist_ok=True, parents=True)

    occupations_file = data_dir / f'occupations-{args.model_key}.json'
    print(f'loading occupations and model predictions from {occupations_file}')
    with occupations_file.open('r') as handle:
        entries = json.load(handle)
    occupations = sorted({entry['occupation'] for entry in entries})
    indexer = {label: index for index, label in enumerate(occupations)}

    representations_file = data_dir / f'occupations-reps-{args.model_key}.pth'
    print(f'loading model entity representations from {representations_file}')
    representations = torch.load(representations_file)
    _, num_layers, hidden_size = representations.shape

    accuracies = []
    for target in ('occupation', 'predictions'):
        for layer in reversed(range(representations.shape[1])):
            print(f'---- probe {target} in layer {layer} ----')

            probe = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, len(occupations)),
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(probe.parameters(), lr=args.lr)
            stopper = training.EarlyStopping(patience=args.patience)

            dataset = [
                # Not a real PyTorch dataset, but who cares?
                {
                    'rep':
                        representations[index, layer],
                    'target':
                        indexer[random.choice(
                            entry[target][:args.model_top_k])]
                        if target == 'predictions' else indexer[entry[target]],
                    **entry,
                } for index, entry in enumerate(entries)
            ]
            train, val = training.random_split(
                dataset,  # type: ignore
                hold_out=args.hold_out)

            train_loader = data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True)
            val_loader = data.DataLoader(val, batch_size=args.batch_size)

            progress_message = 'train probe'
            progress_bar = tqdm(range(args.epochs), desc='train probe')
            best_state_dict = probe.state_dict()
            for epoch in progress_bar:
                probe.train()
                for batch in train_loader:
                    reps = batch['rep'].to(device)
                    targets = batch['target'].to(device)
                    predictions = probe(reps)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                loss = 0.
                probe.eval()
                for batch in val_loader:
                    reps = batch['rep'].to(device)
                    targets = batch['target'].to(device)
                    with torch.inference_mode():
                        predictions = probe(reps)
                    loss += criterion(predictions, targets).item() * len(reps)
                loss /= len(val)

                progress_bar.set_description(
                    f'{progress_message} ({loss:.3f})')

                if stopper(loss):
                    break
                elif loss < stopper.best:
                    best_state_dict = probe.state_dict()

            probe.load_state_dict(best_state_dict)

            # Compute the accuracy on the val set.
            correct = 0
            for batch in val_loader:
                reps = batch['rep'].to(device)
                targets = batch['target'].to(device)
                with torch.inference_mode():
                    logits = probe(reps)
                predictions = logits.topk(k=args.model_top_k, dim=-1).indices
                matches = predictions.eq(targets[:, None]).any(dim=-1)
                correct += matches.sum().item()
            accuracy = correct / len(val)
            print(f'probe val top-{args.model_top_k} accuracy: {accuracy:.3f}')
            accuracies.append({
                'target': target,
                'layer': layer,
                'accuracy': accuracy,
            })

            # Save the probe.
            probe_file = results_dir / f'probe-{target}-layer{layer}.pth'
            print(f'saving probe to {probe_file}')
            torch.save(probe.cpu(), probe_file)

    accuracy_file = results_dir / 'accuracies.json'
    with accuracy_file.open('w') as handle:
        json.dump(accuracies, handle)
