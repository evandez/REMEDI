"""Models for probing state in entity representations."""
from typing import Optional, Tuple, Type, TypeVar, cast

from src.utils import training
from src.utils.typing import Device, StrSequence

import torch
import transformers
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm

ProbingDataset = data.Dataset[Tuple[torch.Tensor, int]]


class AccuracyMixin(nn.Module):
    """Torch module mixin that adds accuracy script."""


class FitMixin(nn.Module):
    """Torch module mixin that adds a simple training script."""

    def fit(
        self,
        dataset: ProbingDataset,
        max_epochs: int = 100,
        batch_size: int = 256,
        hold_out: float = .1,
        lr: float = 1e-3,
        patience: int = 10,
        device: Optional[Device] = None,
        display_progress_as: str = 'train probe',
    ) -> Tuple[float, ...]:
        """Train the probe on the given dataset.

        Args:
            dataset (ProbingDataset): Dataset of reps and targets.
            max_epochs (int, optional): Max unmber of training epochs.
                Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 256.
            hold_out (float, optional): Fraction of data to hold out
                for validation. Defaults to .1.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            patience (int, optional): Stop training if val loss does not
                improve for this many epochs. Defaults to 10.
            device (Optional[Device], optional): Train on this device.
                Defaults to CPU.
            display_progress_as (str, optional): Progress message.
                Defaults to 'train probe'.

        Returns:
            Tuple[float, ...]: Accuracies on train set and held out set.

        """
        if device is not None:
            self.to(device)

        train, val = training.random_split(dataset, hold_out=hold_out)

        train_loader = data.DataLoader(train,
                                       batch_size=batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(val, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        stopper = training.EarlyStopping(patience=patience)

        progress_bar = tqdm(range(max_epochs), desc=display_progress_as)
        best_state_dict = self.state_dict()
        for _ in progress_bar:
            self.train()
            for reps, targets in train_loader:
                if device is not None:
                    reps = reps.to(device)
                    targets = targets.to(device)
                predictions = self(reps)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = 0.
            self.eval()
            for reps, targets in val_loader:
                if device is not None:
                    reps = reps.to(device)
                    targets = targets.to(device)
                with torch.inference_mode():
                    predictions = self(reps)
                loss += criterion(predictions, targets).item() * len(reps)
            loss /= len(val)

            progress_bar.set_description(f'{display_progress_as} ({loss:.3f})')

            if stopper(loss):
                break
            elif loss < stopper.best:
                best_state_dict = self.state_dict()

        self.load_state_dict(best_state_dict)

        accuracies = []
        for key, subset in (('train', train), ('val', val)):
            accuracy = self.accuracy(
                subset,
                batch_size=batch_size,
                device=device,
                display_progress_as=f'compute {key} accuracy')
            accuracies.append(accuracy)

        return tuple(accuracies)

    @torch.inference_mode()
    def accuracy(
        self,
        dataset: ProbingDataset,
        batch_size: int = 256,
        device: Optional[Device] = None,
        display_progress_as: str = 'compute probe accuracy',
    ) -> float:
        """Compute accuracy of probe on dataset.

        Args:
            dataset (ProbingDataset): The probing dataset.
            batch_size (int, optional): Batch size. Defaults to 256.
            device (Optional[Device], optional): Send all data to this device.
                Defaults to None.
            display_progress_as (str, optional): Progress bar message.
                Defaults to 'train probe'.

        Returns:
            float: The accuracy.

        """
        if device is not None:
            self.to(device)

        loader = data.DataLoader(dataset, batch_size=batch_size)

        correct, total = 0, 0
        for reps, targets in tqdm(loader, desc=display_progress_as):
            if device is not None:
                reps = reps.to(device)
                targets = targets.to(device)
            outputs = self(reps)
            predictions = outputs.argmax(dim=-1)
            correct += predictions.eq(targets).sum()
            total += len(reps)
        return correct / total


class MlpProbe(nn.Sequential, FitMixin):
    """A simple two-layer MLP."""

    def __init__(self, input_size: int, num_classes: int):
        """Initialize the probe.

        Args:
            input_size (int): Size of input representations.
            num_classes (int): Number of output classes.

        """
        super().__init__(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, num_classes),
        )
        self.input_size = input_size
        self.num_classes = num_classes


BilinearProbeT = TypeVar('BilinearProbeT', bound='BilinearProbe')


class BilinearProbe(FitMixin):
    """Compare (projected) class reps to (projected) entity reps."""

    def __init__(self, input_size: int, classes: torch.Tensor):
        """Initialize the probe, precomputing class representations.

        Args:
            input_size (int): Size of input representations.
            classes (torch.Tensor): The target class representations.

        """
        super().__init__()
        self.input_size = input_size
        self.register_buffer('classes', classes)
        self.bilinear = nn.Bilinear(input_size, classes.shape[-1], 1)

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        """Compute compatibility between each rep and each class.

        Args:
            reps (torch.Tensor): Input representations. Should have shape
                (batch_size, input_size)

        Returns:
            torch.Tensor: Compatibility scores, with shape
                (batch_size, len(classes)).

        """
        classes = cast(torch.Tensor, self.classes)
        return self.bilinear(
            reps.repeat_interleave(len(classes), dim=0),
            classes.repeat(len(reps), 1),
        )

    @classmethod
    @torch.inference_mode()
    def from_huggingface(
        cls: Type[BilinearProbeT],
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        classes: StrSequence,
        layer: int = -1,
        device: Optional[Device] = None,
    ) -> BilinearProbeT:
        """Initialize the probe for the given huggingface model.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
            model (transformers.PreTrainedModel): The model.
            classes (StrSequence): The class strings.
            layer (int, optional): Layer to take reps from. Defaults to -1.
            device (Optional[Device], optional): Send all data to this device.
                Defaults to None.

        Returns:
            BilinearProbeT: The probe.

        """
        inputs = tokenizer(classes, padding='longest', return_tensors='pt')
        if device is not None:
            inputs = inputs.to(device)
            model.to(device)
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)
        reps = outputs.hidden_states[layer]\
            .mul(outputs.attention_mask[..., None])\
            .sum(dim=1)\
            .div(outputs.attention_mask.sum(dim=-1).view(-1, 1, 1))
        input_size = reps.shape[-1]
        return cls(input_size, reps)
