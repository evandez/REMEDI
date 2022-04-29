"""Probe entity occupations in a discourse."""
import argparse
import json
import pathlib

from src.utils import env

import torch
from torch import cuda
from torch.utils import data
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='probe discourse effect')
    parser.add_argument('--data-dir',
                        type=pathlib.Path,
                        help='read data here (default: project data dir)')
    parser.add_argument(
        '--probe-results-dir',
        type=pathlib.Path,
        help='read trained probes from here (default: project results dir)')
    parser.add_argument(
        '--results-dir',
        type=pathlib.Path,
        help='write results here (default: project results dir)')
    parser.add_argument('--model-key',
                        default='gpt-neo-125M',
                        help='lm to probe (default: gpt-neo-125M)')
    parser.add_argument(
        '--output-key',
        default='probed',
        help='key for probe predictions in output file (default: probed)')
    parser.add_argument('--k',
                        type=int,
                        default=5,
                        help='record top-k probe predictions (default: 5)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        help='probe batch size (default: 256)')
    parser.add_argument('--device', help='device to use (default: guessed)')
    args = parser.parse_args()

    device = args.device or 'cuda' if cuda.is_available() else 'cpu'

    data_dir = args.data_dir or env.data_dir()
    model_data_dir = data_dir / args.model_key

    probe_results_dir = args.probe_results_dir or (env.results_dir() /
                                                   'probe_occupations')
    model_probe_results_dir = probe_results_dir / args.model_key
    if not model_probe_results_dir.exists():
        raise FileNotFoundError(
            f'could not find occupation probes for {args.model_key}; '
            'did you run experiments/probe_occupations.py?')

    results_dir = args.results_dir or env.results_dir() / 'probe_discourse'
    model_results_dir = results_dir / args.model_key
    model_results_dir.mkdir(exist_ok=True, parents=True)

    # Load in original entries so we can update the json.
    discourse_file = data_dir / 'occupations-discourse.json'
    with discourse_file.open('r') as handle:
        entries = json.load(handle)
    for entry in entries:
        entry[args.output_key] = {}

    # Load indexer so we can map probe outputs to strings.
    indexer_file = model_probe_results_dir / 'occupations-indexer.json'
    with indexer_file.open('r') as handle:
        indexer = json.load(handle)
    unindexer = {index: label for label, index in indexer.items()}

    # Load precomputed entity reps and indexer.
    reps = torch.load(model_data_dir / 'occupations-discourse-reps.pth')

    # Compute probe's predictions!
    for layer in range(reps.shape[1]):
        probe_file = (model_probe_results_dir /
                      f'probe-occupation-layer{layer}.pth')
        probe = torch.load(probe_file, map_location=device)
        probe.eval()

        loader = data.DataLoader(data.TensorDataset(reps[:, layer]),
                                 batch_size=args.batch_size)
        predictions = []
        for (batch,) in tqdm(loader, desc=f'probe layer {layer}'):
            with torch.inference_mode():
                outputs = probe(batch.to(device))
                predictions += outputs.topk(k=args.k, dim=-1).indices.tolist()

        assert len(entries) == len(predictions), (len(entries),
                                                  len(predictions))
        for entry, indices in zip(entries, predictions):
            entry[args.output_key][layer] = [unindexer[idx] for idx in indices]

    results_file = model_results_dir / 'occupations-discourse-probed.json'
    with results_file.open('w') as handle:
        json.dump(entries, handle)
