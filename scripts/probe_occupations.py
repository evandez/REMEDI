"""Probe LM representations of an entity for occupation information."""
import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict

from src.utils import env, training_utils

import torch
from torch import cuda, nn, optim
from torch.utils import data
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="probe lm entity reps for occupation")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=("occupation", "predictions"),
        help="target labels to train probes for (default: all)",
    )
    parser.add_argument("--linear", action="store_true", help="use linear probe")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="probe learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="probe batch size (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="probe training epochs (default: 100)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--hold-out",
        type=float,
        default=0.1,
        help="fraction of data to use for val (default: .1)",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=(1, 3, 5),
        help="record top-k accuracies for each k (default: 1 and 3)",
    )
    parser.add_argument(
        "--model-key",
        default="gpt-neo-125M",
        help="lm to probe (default: gpt-neo-125M)",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        help="read data here (default: project data dir)",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        help="write results here (default: project results dir)",
    )
    parser.add_argument("--device", help="device to use (default: guessed)")
    args = parser.parse_args()

    device = args.device or "cuda" if cuda.is_available() else "cpu"

    data_dir = args.data_dir or env.data_dir()
    model_data_dir = data_dir / args.model_key

    results_dir = args.results_dir or env.results_dir() / "probe_occupations"
    model_results_dir = results_dir / args.model_key
    model_results_dir.mkdir(exist_ok=True, parents=True)

    occupations_file = model_data_dir / "occupations-predicted.json"
    print(f"loading occupations and model predictions from {occupations_file}")
    with occupations_file.open("r") as handle:
        entries = json.load(handle)
    occupations = {entry["occupation"] for entry in entries}
    indexer = {label: index for index, label in enumerate(occupations)}

    indexer_file = model_results_dir / "occupations-indexer.json"
    with indexer_file.open("w") as handle:
        json.dump(indexer, handle)

    representations_file = model_data_dir / "occupations-reps.pth"
    print(f"loading model entity representations from {representations_file}")
    representations = torch.load(representations_file)
    _, num_layers, hidden_size = representations.shape

    # Add majority baselines for convenience.
    accuracies = []
    for target in args.targets:
        counts: Dict[str, int] = defaultdict(int)
        for entry in entries:
            label = entry[target]
            if not isinstance(label, str):
                label = label[0]
            counts[label] += 1
        fractions = {label: count / len(entries) for label, count in counts.items()}

        print(f'Class ratios for "{target}":')
        for label, fraction in fractions.items():
            print(f"{label}: {fraction:.3f}")
            accuracies.append(
                {
                    "k": 1,
                    "target": f"predict-{label}",
                    "layer": -1,
                    "accuracy": fraction,
                }
            )

    # Add agreements for convenience.
    for k in args.top_k:
        matching = 0
        for entry in entries:
            matching += entry["occupation"] in entry["predictions"][:k]
        accuracy = matching / len(entries)
        print(f"top-{k} agreement: {accuracy:.3f}")
        accuracies.append(
            {
                "k": k,
                "target": "agreement",
                "layer": -1,
                "accuracy": accuracy,
            }
        )

    # Compute probe accuracies on different targets.
    for target in args.targets:
        for layer in reversed(range(representations.shape[1])):
            print(f'---- probe "{target}" in layer {layer} ----')

            probe: nn.Module
            if args.linear:
                probe = nn.Linear(hidden_size, len(occupations))
            else:
                probe = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_size, len(occupations)),
                )
            probe = probe.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(probe.parameters(), lr=args.lr)
            stopper = training_utils.EarlyStopping(patience=args.patience)

            dataset = [
                # Not a real PyTorch dataset, but who cares?
                {
                    "rep": representations[index, layer],
                    "target": indexer[entry[target][0]]
                    if target == "predictions"
                    else indexer[entry[target]],
                    **entry,
                }
                for index, entry in enumerate(entries)
            ]
            train, val = training_utils.random_split(
                dataset, hold_out=args.hold_out  # type: ignore
            )

            train_loader = data.DataLoader(
                train, batch_size=args.batch_size, shuffle=True
            )
            val_loader = data.DataLoader(val, batch_size=args.batch_size)

            progress_message = "train probe"
            progress_bar = tqdm(range(args.epochs), desc="train probe")
            best_state_dict = probe.state_dict()
            for epoch in progress_bar:
                probe.train()
                for batch in train_loader:
                    reps = batch["rep"].to(device)
                    targets = batch["target"].to(device)
                    predictions = probe(reps)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                loss = 0.0
                probe.eval()
                for batch in val_loader:
                    reps = batch["rep"].to(device)
                    targets = batch["target"].to(device)
                    with torch.inference_mode():
                        predictions = probe(reps)
                    loss += criterion(predictions, targets).item() * len(reps)
                loss /= len(val)

                progress_bar.set_description(f"{progress_message} ({loss:.3f})")

                if stopper(loss):
                    break
                elif loss < stopper.best:
                    best_state_dict = probe.state_dict()

            probe.load_state_dict(best_state_dict)

            # Compute the accuracy on the val set.
            for k in args.top_k:
                correct = 0
                for batch in val_loader:
                    reps = batch["rep"].to(device)
                    targets = batch["target"].to(device)
                    with torch.inference_mode():
                        logits = probe(reps)
                    predictions = logits.topk(k=k, dim=-1).indices
                    matches = predictions.eq(targets[:, None]).any(dim=-1)
                    correct += matches.sum().item()
                accuracy = correct / len(val)
                print(f"probe val top-{k} accuracy: {accuracy:.3f}")
                accuracies.append(
                    {
                        "top-k": k,
                        "target": target,
                        "layer": layer,
                        "accuracy": accuracy,
                    }
                )

            # Save the probe.
            probe_file = model_results_dir / f"probe-{target}-layer{layer}.pth"
            print(f"saving probe to {probe_file}")
            torch.save(probe.cpu(), probe_file)

    accuracy_file = model_results_dir / "accuracies.json"
    with accuracy_file.open("w") as handle:
        json.dump(accuracies, handle)
