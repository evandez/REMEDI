"""Editing models."""
from dataclasses import dataclass
from typing import Optional, cast

from src import precompute
from src.utils import dataset_utils, model_utils, training_utils
from src.utils.typing import Dataset, Device

import datasets
import torch
import torch.utils.data
from baukit import nethook
from torch import nn, optim
from tqdm.auto import tqdm


def _editing_loss(
    *,
    mt: model_utils.ModelAndTokenizer,
    layer: int,
    editor: "Editor",
    batch: dict,
    lam: float = 0.25,
    kl: Optional[nn.KLDivLoss] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Apply the edit to the representat

    See `src.precompute.editor_inputs_from_dataset` for expected format of `batch`.
    """
    [layer_path] = model_utils.determine_layer_paths(mt, layers=[layer])
    prompt = batch["prompt"]
    entity_i, entity_j = batch["prompt.token_range.entity"]
    attr_i, attr_j = batch["context.token_range.attribute"]
    hiddens_context = batch[f"context.hiddens.{layer}"].to(device)
    hiddens_attr_avg = hiddens_context[None, attr_i:attr_j].mean(dim=1)

    inputs = mt.tokenizer(
        prompt, return_tensors="pt", padding="longest", truncate=True
    ).to(device)

    # If necessary, determine original next token distribution.
    logps_orig = None
    if kl is not None:
        with torch.inference_mode():
            outputs = mt.model(**inputs)
            logps_orig = torch.log_softmax(outputs.logits, dim=-1)

    def edit_output(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Apply the edit to the representation."""
        direction = editor(hiddens_attr_avg)
        for bi, (i, j) in enumerate(zip(entity_i, entity_j)):
            output[0][bi, i:j] = output[0][bi, i:j] + direction[bi]
        return (output[0], *output[1:])

    # Compute token distribution after the edit.
    with nethook.Trace(mt.model, layer=layer_path, edit_output=edit_output):
        outputs = mt.model(**inputs)
    logps_edit = torch.log_softmax(outputs.logits, dim=-1)

    # Compute simple loss: the probability of the target token post-edit.
    loss = torch.tensor(0.0, device=device)
    indices = inputs.attention_mask.sum(dim=-1) - 1
    for bi, (si, mti) in enumerate(zip(indices.tolist(), batch["mediated_token_id"])):
        logp_mediated = logps_edit[bi, si, mti]
        loss += -logp_mediated
    batch_size = indices.shape[0]
    loss /= batch_size

    # If specified, include a KL loss term with the original token distribution.
    if kl is not None:
        assert logps_orig is not None
        logps_edit = logps_edit[torch.arange(batch_size), indices]
        logps_orig = logps_orig[torch.arange(batch_size), indices]
        loss += lam * kl(logps_edit, logps_orig)

    return loss


@dataclass(frozen=True)
class FitRun:
    """Results from running `Editor.fit`."""

    dataset: Dataset


class Editor(nn.Module):
    """A simple linear editing model."""

    def __call__(self, attribute: torch.Tensor) -> torch.Tensor:
        """Map the attribute hidden representation to an entity edit direction."""
        raise NotImplementedError

    def fit(
        self,
        *,
        mt: model_utils.ModelAndTokenizer,
        dataset: Dataset,
        layer: int,
        max_epochs: int = 10,
        editor_batch_size: int = 256,
        model_batch_size: int = 1,
        hold_out: float = 0.1,
        lr: float = 1e-2,
        patience: int = 4,
        device: Optional[Device] = None,
        assume_inputs_precomputed: bool = False,
    ) -> FitRun:
        """Train this editor.

        Args:
            mt: The model and tokenizer.
            dataset: Any context mediation dataset. If a DatasetDict, must have train
                and val splits.
            layer: Layer to perform edits at.
            max_epochs: Max epochs to train for.
            editor_batch_size: Batch size for training editor.
            model_batch_size: Batch size for precomputing model states.
            hold_out: Hold out this fraction of data for validation.
            lr: Learning rate. Defaults to 1e-2.
            patience: Stop after val loss does not improve for this many epochs.
            device: Run editor and model on this device.
            assume_inputs_precomputed: If set, assume dataset already has
                precomputed inputs for the editor.

        """
        dataset = dataset_utils.maybe_train_test_split(dataset, test_size=hold_out)
        if not assume_inputs_precomputed:
            dataset = precompute.editor_inputs_from_dataset(
                mt,
                dataset,
                layers=[layer],
                precompute_hiddens_batch_size=model_batch_size,
                device=device,
            )

        mt.model.to(device)
        self.to(device)

        mt.model.eval()
        for parameter in mt.model.parameters():
            parameter.requires_grad_(True)

        optimizer = optim.AdamW(self.parameters(), lr=lr)
        stopper = training_utils.EarlyStopping(patience=patience)

        with dataset.formatted_as("torch"):
            train = cast(torch.utils.data.Dataset, dataset["train"])
            val = cast(torch.utils.data.Dataset, dataset["test"])
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=editor_batch_size
            )
            val_loader = torch.utils.data.DataLoader(val, batch_size=editor_batch_size)

            # TODO(evandez): More flexible loss function.
            kl = nn.KLDivLoss(reduction="batchmean", log_target=True).to(device)

            best = self.state_dict()
            progress_bar = tqdm(range(max_epochs), desc="train editor")
            for epoch in progress_bar:
                self.train()
                train_loss = 0.0
                for batch in train_loader:
                    loss = _editing_loss(
                        mt=mt,
                        layer=layer,
                        editor=self,
                        batch=batch,
                        kl=kl,
                        device=device,
                    )
                    if epoch > 0:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                self.eval()
                val_loss = 0.0
                for batch in val_loader:
                    val_loss += _editing_loss(
                        mt=mt,
                        layer=layer,
                        editor=self,
                        batch=batch,
                        kl=kl,
                        device=device,
                    ).item()
                val_loss /= len(val_loader)

                progress_bar.set_description(
                    f"train {train_loss:.3f} / val {val_loss:.3f}"
                )
                if stopper(val_loss):
                    break
                elif stopper.improved:
                    best = self.state_dict()

        self.load_state_dict(best)
        return FitRun(dataset=dataset)


class LinearEditor(Editor):
    """A simple linear model, optionally with a rank constraint."""

    def __init__(self, mt: model_utils.ModelAndTokenizer, rank: Optional[int] = None):
        """Initialize the editor.

        Args:
            rank: Rank constraint on the linear transformation. Defaults to None.

        """
        super().__init__()

        hidden_size = model_utils.determine_hidden_size(mt)

        self.w: nn.Linear | nn.Sequential
        if rank is None:
            self.w = nn.Linear(hidden_size, hidden_size)
        else:
            self.w = nn.Sequential(
                nn.Linear(hidden_size, rank),
                nn.Linear(rank, hidden_size),
            )

    def __call__(self, attribute: torch.Tensor) -> torch.Tensor:
        """Compute the edit direction."""
        return self.w(attribute)
