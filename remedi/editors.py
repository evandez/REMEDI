"""Editing models."""
import argparse
import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, DefaultDict, Literal, Optional, cast

from remedi import data, models, precompute
from remedi.utils import tokenizer_utils, training_utils
from remedi.utils.typing import (
    Dataset,
    Device,
    Model,
    ModelGenerateOutput,
    ModelOutput,
    PathLike,
    Tokenizer,
)

import torch
import torch.utils.data
import transformers.modeling_outputs
from baukit import nethook, runningstats
from dataclasses_json import DataClassJsonMixin
from torch import nn, optim
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_LAM_M = 1.0
DEFAULT_LAM_U = 1.0
DEFAULT_LAM_KL = 10.0
DEFAULT_N_TOP = 10
DEFAULT_MAX_LENGTH = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_EPOCHS = 20
DEFAULT_PATIENCE = 2
DEFAULT_LR = 1e-3
DEFAULT_HOLD_OUT = 0.1


class apply_direction(contextlib.AbstractContextManager):
    """Context manager that adds directions to specific hiddens during forward pass.

    Args:
        model: The model to apply the directions to.
        layer: The layer to apply the directions at.
        directions: Directions to apply. Needs shape (batch_size, hidden_size).
        token_ranges: Token ranges to apply direction at. Needs shape (batch_size, 2).
        alpha: Weight of edit direction when applying edit direction.
        beta: Weight of entity token when applying edit direction.

    """

    def __init__(
        self,
        *,
        model: Model,
        layer: int,
        directions: torch.Tensor,
        token_ranges: torch.Tensor,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ):
        """Initialize the context manager."""
        self.model = model
        self.layer = layer
        self.directions = directions
        self.token_ranges = token_ranges
        self.alpha = alpha
        self.beta = beta
        self._trace = None

    def __enter__(self) -> Model:
        """Add a hook to the model so the direction is applied."""
        [layer_path] = models.determine_layer_paths(self.model, layers=[self.layer])

        def edit_output(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            """Apply the directions."""
            if output[0].shape[1] == 1:
                # This condition is satisfied only when we're generating beyond the
                # prompt, in which case, we never want to edit the output.
                return output
            for bi, (i, j) in enumerate(self.token_ranges.tolist()):
                output[0][bi, i:j] = (
                    self.beta * output[0][bi, i:j] + self.alpha * self.directions[bi]
                )
            return (output[0], *output[1:])

        self._trace = nethook.Trace(
            self.model, layer=layer_path, edit_output=edit_output
        )
        return self.model

    def __exit__(self, *_: Any) -> Optional[bool]:
        """Unhook the model."""
        assert self._trace is not None
        self._trace.close()
        self._trace = None
        return None


@dataclass(frozen=True)
class EditedModelOutput:
    """Wrapper around a run of an edited model."""

    direction: torch.Tensor
    token_range: torch.Tensor
    output: ModelOutput | ModelGenerateOutput


class EditedModel(nn.Module):
    """Wrapper around an LM which knows how to automatically apply mediation.

    Importantly, this model assumes inputs look like `ContextMediationSample`s
    or batches of them. See `src/data.py` for more details.
    """

    def __init__(
        self,
        editor: "Editor",
        mt: Optional[models.ModelAndTokenizer] = None,
        layer: Optional[int] = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        device: Optional[Device] = None,
    ):
        """Wrap the model to be edited."""
        super().__init__()
        self.mt = mt if mt is not None else editor.mt
        self.layer = layer if layer is not None else editor.layer
        self.editor = editor
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def maybe_compute_editor_inputs(
        self,
        batch: data.ContextMediationInput | dict,
        entity_occurrence: int = 0,
    ) -> dict:
        """Maybe compute hidden states for batch, if not already present."""
        precomputed = {**batch}
        if not precompute.has_editor_inputs(precomputed):
            precomputed = precompute.editor_inputs_from_batch(
                self.mt,
                cast(data.ContextMediationInput, batch),
                layers=[self.layer],
                device=self.device,
                return_target_token_ids=False,
                entity_occurrence_in_prompt=entity_occurrence,
            )
        return precomputed

    def compute_edit_directions(self, batch: dict, **kwargs: Any) -> torch.Tensor:
        """Compute edit directions for batch."""
        precomputed = self.maybe_compute_editor_inputs(batch, **kwargs)

        hiddens_entity_key = f"entity.entity.hiddens.{self.editor.layer}."
        if self.editor.input_last_entity_token:
            hiddens_entity_key += "last"
        else:
            hiddens_entity_key += "average"

        hiddens_entity = precomputed[hiddens_entity_key]
        hiddens_attr = precomputed[
            f"context.attribute.hiddens.{self.editor.layer}.average"
        ]

        dtype = self.mt.model.config.torch_dtype
        hiddens_entity = cast(torch.Tensor, hiddens_entity).to(self.device, dtype)
        hiddens_attr = cast(torch.Tensor, hiddens_attr).to(self.device, dtype)

        directions = self.editor(entity=hiddens_entity, attribute=hiddens_attr)

        return directions

    def compute_model_outputs(
        self,
        batch: data.ContextMediationInput | dict,
        generate: bool = False,
        inputs: Optional[transformers.BatchEncoding] = None,
        padding_side: Optional[str] = None,
        entity_occurrence: int = 0,
        **kwargs: Any,
    ) -> EditedModelOutput:
        """Run the model on the inputs, editing its hidden reps in the process."""
        precomputed = self.maybe_compute_editor_inputs(
            batch, entity_occurrence=entity_occurrence
        )
        directions = self.compute_edit_directions(precomputed)

        # Which token we apply the edit to depends on which side padding is applied. If
        # caller tells us which, use that, but otherwise just infer from the tokenizer.
        if padding_side is None:
            padding_side = self.mt.tokenizer.padding_side

        if padding_side == "right":
            token_range_key = "token_range"
        else:
            token_range_key = "negative_token_range"

        entity_ij_key = f"prompt.entity.{token_range_key}"
        if self.editor.edit_last_entity_token:
            entity_ij_key += ".last"

        entity_ij = cast(torch.Tensor, precomputed[entity_ij_key])

        # Now do forward pass on edited model.
        if inputs is None:
            prompt = batch["prompt"]
            inputs, _ = precompute.inputs_from_batch(
                self.mt, prompt, device=self.device
            )
        with apply_direction(
            model=self.mt.model,
            layer=self.layer,
            directions=directions,
            token_ranges=entity_ij,
            alpha=self.alpha,
            beta=self.beta,
        ) as model:
            if generate:
                if "max_new_tokens" not in kwargs:
                    kwargs.setdefault("max_length", DEFAULT_MAX_LENGTH)
                kwargs.setdefault("pad_token_id", self.mt.tokenizer.eos_token_id)
                outputs = model.generate(**inputs, **kwargs)
            else:
                outputs = model(**inputs, **kwargs)

        return EditedModelOutput(
            direction=directions,
            token_range=entity_ij,
            output=outputs,
        )

    def forward(
        self,
        batch: data.ContextMediationInput,
        inputs: Optional[transformers.BatchEncoding] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Compute the edited outputs for the context mediation sample.

        Args:
            batch: The sample or batch of samples.
            inputs: Precomputed tokenized prompt. If not set, will read
                `batch["prompt"]` and pass it through the tokenizer.

        Returns:
            Standard huggingface outputs, but for the edited model.

        """
        edit = self.compute_model_outputs(
            batch, inputs=inputs, generate=False, **kwargs
        )
        return cast(ModelOutput, edit.output)

    def generate(
        self,
        batch: data.ContextMediationInput,
        inputs: Optional[transformers.BatchEncoding] = None,
        **kwargs: Any,
    ) -> ModelGenerateOutput:
        """Forward to `mt.model.generate`, but apply editor."""
        edit = self.compute_model_outputs(batch, inputs=inputs, generate=True, **kwargs)
        return cast(ModelGenerateOutput, edit.output)


@dataclass(frozen=True)
class EditedModelAndTokenizer:
    """Counterpart to ModelAndTokenizer for edited models."""

    model: EditedModel
    tokenizer: Tokenizer


class apply(contextlib.AbstractContextManager):
    """Context manager that computes and applies edit directions in place.

    Args:
        editor: The editor to apply.
        mt: The model/tokenizer to apply the editor to. By default, uses the model
            that the editor was trained, but you could (in theory) specify any
            model which has the same number of layers!
        alpha: Step size to take when applying the direction.

    """

    def __init__(
        self,
        editor: "Editor",
        mt: Optional[models.ModelAndTokenizer] = None,
        layer: Optional[int] = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        device: Optional[Device] = None,
    ):
        """Initialize the context manager."""
        self.mt = mt
        self.layer = layer
        self.editor = editor
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self._hooked: EditedModel | None = None

    def __enter__(
        self,
    ) -> EditedModelAndTokenizer:
        """Wrap the model."""
        self._hooked = EditedModel(
            self.editor,
            mt=self.mt,
            layer=self.layer,
            alpha=self.alpha,
            beta=self.beta,
            device=self.device,
        )
        return EditedModelAndTokenizer(self._hooked, self._hooked.mt.tokenizer)

    def __exit__(self, *_: Any) -> Optional[bool]:
        """Unwrap the model."""
        self._hooked = None
        return None


def editing_loss(
    *,
    editor: "Editor",
    batch: dict,
    lam_m: float | None = DEFAULT_LAM_M,
    lam_u: float | None = DEFAULT_LAM_U,
    lam_kl: float | None = DEFAULT_LAM_KL,
    lam_norm: float | None = None,
    lam_ess: float | None = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Apply edits and compute the loss.

    See `src.precompute.editor_inputs_from_dataset` for expected format of `batch`.
    """
    prompt = batch["prompt"]

    inputs, _ = precompute.inputs_from_batch(editor.mt, prompt, device=device)
    with apply(editor, device=device) as mt_edit:
        edit = mt_edit.model.compute_model_outputs(batch, inputs=inputs)
    outputs_edit = edit.output
    logp_edit = torch.log_softmax(outputs_edit.logits, dim=-1)

    batch_size = len(prompt)
    batch_idx = torch.arange(batch_size)
    last_idx = precompute.last_token_index_from_batch(inputs)

    # Loss is a sum of different terms. Which terms depends on the experiment.
    loss = logp_edit.new_zeros(1)

    # Most basic term, almost always included: probability of the target token.
    if lam_m is not None:
        mediated_token_ids = batch.get("target_mediated.token_id")
        if mediated_token_ids is None:
            raise ValueError("lam_m > 0 but target_mediated not found?")
        loss -= lam_m * logp_edit[batch_idx, last_idx, mediated_token_ids].mean()

    # If requested, penalize probability mass on the unmediated token.
    if lam_u is not None:
        unmediated_token_ids = batch.get("target_unmediated.token_id")
        if unmediated_token_ids is None:
            raise ValueError("lam_u > 0 but target_unmediated not found?")
        logp_edit_unmediated = logp_edit[batch_idx, last_idx, unmediated_token_ids]
        p_edit_unmediated = torch.exp(logp_edit_unmediated)
        loss_unmediated = torch.log(1 - p_edit_unmediated).mean()
        loss -= lam_u * loss_unmediated

    # If requested, apply KL term to tokens between entity and next token.
    if lam_kl is not None:
        between_ijs = torch.empty(batch_size, 2, dtype=torch.long)
        between_ijs[:, 0] = batch["prompt.entity.token_range"][:, -1] - 1
        between_ijs[:, 1] = batch["prompt.length"] - 1

        with torch.inference_mode():
            outputs_orig = editor.mt.model(**inputs)
        logp_orig = torch.log_softmax(outputs_orig.logits, dim=-1)

        loss_kl = torch.zeros_like(loss)
        for bi, (si, sj) in enumerate(between_ijs.tolist()):
            assert si >= 0 and sj >= 0
            if sj <= si:
                continue
            logp_edit_between = logp_edit[bi, si:sj].view(-1, logp_edit.shape[-1])
            logp_orig_between = logp_orig[bi, si:sj].view(-1, logp_orig.shape[-1])
            loss_kl = nn.functional.kl_div(
                logp_edit_between, logp_orig_between, reduction="sum", log_target=True
            )
        loss_kl /= batch_size
        loss += lam_kl * loss_kl

    # If requested, penalize the norm of the resulting directions.
    if lam_norm is not None:
        loss += lam_norm * edit.direction.norm(p=2, dim=-1).mean()

    # If requested, penalize changes to an essence prompt.
    if lam_ess is not None:
        entities = batch["entity"]
        prompts_ess = [f"{entity} is a" for entity in entities]

        inputs_ess, _ = precompute.inputs_from_batch(
            editor.mt, prompts_ess, device=device
        )
        with torch.inference_mode():
            outputs_before = editor.mt.model(**inputs_ess)
        with apply(editor, device=device) as mt_edit:
            outputs_after = mt_edit.model(
                {
                    "prompt": prompts_ess,
                    "entity": entities,
                    "context": batch["context"],
                    "attribute": batch["attribute"],
                }
            )

        last_idx_ess = precompute.last_token_index_from_batch(inputs_ess)
        logits_edit_ess: torch.Tensor = outputs_after.logits[batch_idx, last_idx_ess]
        logits_orig_ess: torch.Tensor = outputs_before.logits[batch_idx, last_idx_ess]

        logp_edit_ess = torch.log_softmax(logits_edit_ess, dim=-1)
        logp_edit_orig = torch.log_softmax(logits_orig_ess, dim=-1)

        loss_ess = nn.functional.kl_div(
            logp_edit_ess,
            logp_edit_orig,
            log_target=True,
            reduction="batchmean",
        )
        loss += lam_ess * loss_ess

    return loss


@dataclass(frozen=True)
class EditorTrainingRun:
    """Results from running `Editor.fit`."""

    dataset: Dataset


@dataclass(frozen=True)
class EditorEvaluationResult(DataClassJsonMixin):
    """Result for single sample from `Editor.evaluate`."""

    sample: dict

    # NOTE(evandez): Need explicit types here so DataClassJsonMixin works.
    before_top_tokens: list[str] | None = None
    before_top_logps: list[float] | None = None
    before_generations: list[str] | None = None

    after_top_tokens: list[str] | None = None
    after_top_logps: list[float] | None = None
    after_generations: list[str] | None = None

    before_target_mediated_score: float | None = None
    before_target_unmediated_score: float | None = None

    after_target_mediated_score: float | None = None
    after_target_unmediated_score: float | None = None


@dataclass(frozen=True)
class EditorEvaluateRun(DataClassJsonMixin):
    """Wrapper around a list of individual evaluation results."""

    results: list[EditorEvaluationResult]


@dataclass(frozen=True)
class EditorClassificationResult(DataClassJsonMixin):
    """Result for single sample from `Editor.classify`."""

    sample: dict[str, str]

    # Scores computed with entity and editor direction.
    score_mediated: float
    score_unmediated: float


@dataclass(frozen=True)
class EditorClassifyRun(DataClassJsonMixin):
    """Wrapper around a list of individual classification results."""

    results: list[EditorClassificationResult]


class Editor(nn.Module):
    """Abstract base clase for editing models."""

    def __init__(
        self,
        *,
        mt: models.ModelAndTokenizer,
        layer: int,
        input_last_entity_token: bool = True,
        edit_last_entity_token: bool = True,
    ):
        """Initialize the editor."""
        super().__init__()
        self.mt = mt
        self.layer = layer
        self.input_last_entity_token = input_last_entity_token
        self.edit_last_entity_token = edit_last_entity_token

    def to_(self, mt: models.ModelAndTokenizer | None = None) -> None:
        """Make this editor's device/dtype match the underlying models."""
        mt = self.mt if mt is None else mt
        self.to(
            device=models.determine_device(mt),
            dtype=models.determine_dtype(mt),
        )

    def forward(self, *, entity: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
        """Map the attribute hidden representation to an entity edit direction."""
        raise NotImplementedError

    def fit(
        self,
        *,
        dataset: Dataset,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hold_out: float = DEFAULT_HOLD_OUT,
        lr: float = DEFAULT_LR,
        lam_m: float | None = DEFAULT_LAM_M,
        lam_u: float | None = DEFAULT_LAM_U,
        lam_kl: float | None = DEFAULT_LAM_KL,
        lam_norm: float | None = None,
        lam_ess: float | None = None,
        patience: int = DEFAULT_PATIENCE,
        device: Optional[Device] = None,
    ) -> EditorTrainingRun:
        """Train this editor.

        Args:
            mt: The model and tokenizer.
            dataset: Any context mediation dataset. If a DatasetDict, must have train
                and val splits.
            max_epochs: Max epochs to train for.
            batch_size: Batch size for training editor. Note this is constrained mostly
                by how many sentences the model can process at once!
            hold_out: Hold out this fraction of data for validation.
            lr: Learning rate.
            lam_m: Loss weight for p(mediated) term.
            lam_u: Loss weight for log[1 - p(unmediated)] term.
            lam_kl: Loss weight for KL div on token distributions between entity
                and attribute ine prompt.
            lam_norm: Loss weight for norm of predicted directions.
            lam_ess: Loss weight for essence penalty.
            patience: Stop after val loss does not improve for this many epochs.
            device: Run editor and model on this device.

        """
        dataset = data.maybe_train_test_split(dataset, test_size=hold_out)

        self.mt.model.to(device)
        self.to(device)

        self.mt.model.eval()
        for parameter in self.mt.model.parameters():
            parameter.requires_grad_(True)

        # NOTE(evandez): If we're using half precision, we have to increase the eps
        # used by Adam or we'll immediately get NaNs everywhere. It's because the
        # default eps gets rounded to 0 in half precision.
        optimizer_kwargs: dict = dict(lr=lr)
        if models.determine_dtype(self.mt) is torch.float16:
            optimizer_kwargs["eps"] = 1e-4

        optimizer = optim.AdamW(self.parameters(), **optimizer_kwargs)
        stopper = training_utils.EarlyStopping(patience=patience)

        # Since sometimes these fields are None, exclude them from the dataset
        # so the DataLoader does not freak out.
        exclude_columns = ["source"]
        if lam_u is None:
            exclude_columns.append("target_unmediated")
        if lam_m is None:
            exclude_columns.append("target_mediated")
        columns = data.column_names(dataset, exclude=exclude_columns)

        with dataset.formatted_as("torch", columns=columns):
            train = cast(torch.utils.data.Dataset, dataset["train"])
            val = cast(torch.utils.data.Dataset, dataset["test"])
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
            val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)

            best = self.state_dict()
            for epoch in range(max_epochs + 1):
                desc = f"epoch {epoch}/{max_epochs}"

                self.train()
                train_loss = 0.0
                train_progress_bar = tqdm(train_loader)
                for batch in train_progress_bar:
                    optimizer.zero_grad()
                    loss = editing_loss(
                        editor=self,
                        batch=batch,
                        lam_m=lam_m,
                        lam_u=lam_u,
                        lam_kl=lam_kl,
                        lam_norm=lam_norm,
                        lam_ess=lam_ess,
                        device=device,
                    )
                    if epoch > 0:
                        loss.backward()
                        optimizer.step()
                    train_loss += loss.item()
                    train_progress_bar.set_description(
                        f"{desc} train={loss.item():.2f}"
                    )
                train_loss /= len(train_loader)

                self.eval()
                val_loss = 0.0
                val_progress_bar = tqdm(val_loader)
                for batch in val_progress_bar:
                    with torch.inference_mode():
                        loss = editing_loss(
                            editor=self,
                            batch=batch,
                            lam_u=lam_u,
                            lam_m=lam_m,
                            lam_kl=lam_kl,
                            lam_norm=lam_norm,
                            lam_ess=lam_ess,
                            device=device,
                        )
                    val_loss += loss.item()
                    val_progress_bar.set_description(
                        f"{desc} val={loss.item():.2f} best={stopper.best:.2f}"
                    )
                val_loss /= len(val_loader)

                if stopper(val_loss):
                    break
                elif stopper.improved:
                    best = self.state_dict()

        self.load_state_dict(best)
        return EditorTrainingRun(dataset=dataset)

    @torch.inference_mode()
    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_top: int = DEFAULT_N_TOP,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        desc: Optional[str] = None,
        device: Optional[Device] = None,
        return_before: bool = True,
        return_after: bool = True,
        return_mediated: bool = True,
        return_unmediated: bool = True,
    ) -> EditorEvaluateRun:
        """Evaluate the editor on a held out set.

        Note this function does *not* compute metrics for the editor. For those, see
        `scripts/benchmark.py`. Instead, this function computes post-edit generations
        on the given dataset. Useful for visualizing edit results during and after
        training while not committing to running the whole benchmark.

        Args:
            dataset: Dataset to evaluate on.
            batch_size: Model batch size.
            n_top: Number of top words/probs to return.
            max_length: Number of tokens to generate including prompt.
            max_new_tokens: Number of tokens to generate not including prompt.
            top_k: Value of k for top-k sampling, which is always used to generate.
            alpha: Weight of edit direction when applying edit direction.
            beta: Weight of entity token when applying edit direction.
            desc: The tqdm description.
            device: Send all data to this device. Defaults to None.
            return_before: Return model predictions before edit.
            return_after: Return model predictions after edit.
            return_mediated: Return mediated token probability.
            return_unmediated: Return unmediated token probability.

        Returns:
            The evaluation results, one per entry in the dataset.

        """
        self.mt.eval_()
        self.mt.to_(device)
        self.eval().to(device)
        include_target_probs = "target_mediated" in dataset.column_names
        if desc is None:
            desc = f"evaluate editor (layer={self.layer})"
        if max_length is None and max_new_tokens is None:
            max_length = DEFAULT_MAX_LENGTH

        # Remove risk of accessing a null column if we do not need to.
        exclude_columns = []
        if not return_mediated:
            exclude_columns.append("target_mediated")
        if not return_unmediated:
            exclude_columns.append("target_unmediated")
        columns = data.column_names(dataset, exclude=exclude_columns)

        results = []
        with dataset.formatted_as("torch", columns=columns):
            loader = torch.utils.data.DataLoader(
                cast(torch.utils.data.Dataset, dataset), batch_size=batch_size
            )
            for batch in tqdm(loader, desc=desc):
                if not precompute.has_editor_inputs(batch):
                    batch.update(
                        precompute.editor_inputs_from_batch(
                            mt=self.mt,
                            batch=batch,
                            layers=[self.layer],
                            device=device,
                            return_entity_hiddens=return_after,
                            return_attribute_hiddens=return_after,
                            return_target_token_ids=return_mediated
                            or return_unmediated,
                        )
                    )

                prompts = batch["prompt"]
                current_batch_size = len(prompts)

                # Need padding side to be left for batch generate.
                with models.set_padding_side(self.mt, padding_side="left"):
                    inputs, _ = precompute.inputs_from_batch(
                        self.mt, prompts, device=device
                    )

                generate_kwargs = dict(
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.mt.tokenizer.eos_token_id,
                )
                if top_k is not None:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["top_k"] = top_k
                if max_length is not None:
                    generate_kwargs["max_length"] = max_length
                if max_new_tokens is not None:
                    generate_kwargs["max_new_tokens"] = max_new_tokens

                outputs_before = None
                if return_before:
                    outputs_before = self.mt.model.generate(**inputs, **generate_kwargs)

                outputs_after = None
                if return_after:
                    with apply(
                        self, alpha=alpha, beta=beta, device=device
                    ) as edited_mt:
                        outputs_after = edited_mt.model.generate(
                            batch, inputs=inputs, padding_side="left", **generate_kwargs
                        )

                batched_results: dict = {}
                for key, outputs in (
                    ("before", outputs_before),
                    ("after", outputs_after),
                ):
                    if outputs is None:
                        continue

                    first_token_logps = torch.log_softmax(outputs.scores[0], dim=-1)
                    top_logps, top_token_ids = first_token_logps.topk(k=n_top, dim=-1)
                    top_tokens = tokenizer_utils.batch_convert_ids_to_tokens(
                        top_token_ids, self.mt.tokenizer
                    )
                    generations = self.mt.tokenizer.batch_decode(
                        outputs.sequences, skip_special_tokens=True
                    )

                    batched_results[f"{key}_top_logps"] = top_logps.tolist()
                    batched_results[f"{key}_top_tokens"] = top_tokens
                    batched_results[f"{key}_generations"] = [[g] for g in generations]

                    if include_target_probs:
                        target_keys = []
                        if return_mediated:
                            target_keys.append("mediated")
                        if return_unmediated:
                            target_keys.append("unmediated")
                        batch_indices = torch.arange(current_batch_size)
                        for target_key in target_keys:
                            target_id = batch[f"target_{target_key}.token_id"]
                            target_probs = first_token_logps[batch_indices, target_id]
                            target_prob_key = f"{key}_target_{target_key}_score"
                            batched_results[target_prob_key] = target_probs.tolist()

                for bi in range(current_batch_size):
                    result: dict = {k: vs[bi] for k, vs in batched_results.items()}
                    results.append(result)

        # Finally, decorate results with original sample data.
        assert len(results) == len(dataset)
        for sample, result in zip(dataset, results):
            result.update(
                sample={
                    key: sample[key]
                    for key in data.ContextMediationSample.__required_keys__
                }
            )

        return EditorEvaluateRun([EditorEvaluationResult(**r) for r in results])

    @torch.inference_mode()
    def classify(
        self,
        *,
        dataset: Dataset,
        normalize: bool = True,
        cosine: bool = False,
        take_entity_from: Literal["prompt_in_context", "prompt", "entity"] = "entity",
        entity_layer: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Device | None = None,
        desc: str | None = "classify",
    ) -> EditorClassifyRun:
        """Determine whether model believes unmediated/mediated attr is true of entity.

        Uses cosine similarity of edit direction with entity representation as a
        measure of how much model believes attribute to be true of entity.

        Args:
            dataset: The datast to run on.
            normalize: Normalize hiddens to have 0 mean and unit variance.
            cosine: Use cosine similarity instead of dot product.
            take_entity_from: Which prompt to take entity representation from.
            entity_layer: The layer to get the entity rep from. This can be different
                from the edit layer!
            batch_size: Model batch size.
            device: Send model and data to this device.
            desc: Progress bar description.

        Returns:
            The classification results for each sample.

        """
        self.mt.eval_()
        self.mt.to_(device)
        self.eval().to(device)

        dtype = models.determine_dtype(self.mt)

        sim: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        if cosine:
            if dtype is torch.float16:
                sim = training_utils.cosine_similarity_float16
            else:
                sim = torch.nn.functional.cosine_similarity
        else:
            sim = torch.dot

        if entity_layer is None:
            entity_layer = self.layer
        layers = sorted({self.layer, entity_layer})

        key_e = f"{take_entity_from}.entity.hiddens.{entity_layer}.last"
        key_u = f"context_unmediated.attribute_unmediated.hiddens.{self.layer}.average"
        key_m = f"context.attribute.hiddens.{self.layer}.average"

        # First the expensive part: compute directions for all entities and attributes.
        with dataset.formatted_as("torch"):
            loader = torch.utils.data.DataLoader(
                cast(torch.utils.data.Dataset, dataset), batch_size=batch_size
            )
            directions = []
            for batch in tqdm(loader, desc=desc):
                if not precompute.has_classification_inputs(batch):
                    batch.update(
                        precompute.classification_inputs_from_batch(
                            self.mt, batch, layers=layers, device=device
                        )
                    )

                entities = batch[key_e].to(device, dtype)
                attributes_u = batch[key_u].to(device, dtype)
                attributes_m = batch[key_m].to(device, dtype)

                directions_u = self(entity=entities, attribute=attributes_u)
                directions_m = self(entity=entities, attribute=attributes_m)

                for bi in range(len(entities)):
                    directions.append(
                        dict(
                            entity=entities[bi],
                            unmediated=directions_u[bi],
                            mediated=directions_m[bi],
                        )
                    )

        # Then, if necessary, normalize the directions.
        if normalize:
            for key in ("entity", "unmediated", "mediated"):
                values = torch.stack([direction[key] for direction in directions])
                mu, std = values.mean(dim=0), values.std(dim=0)
                for direction in directions:
                    direction[key] = (direction[key] - mu) / std

        # Finally, compute the alignment scores.
        assert len(directions) == len(dataset)
        results = []
        for sample, direction in zip(dataset, directions):
            h_e = direction["entity"]
            h_u = direction["unmediated"]
            h_m = direction["mediated"]
            score_u = sim(h_e, h_u)
            score_m = sim(h_e, h_m)
            results.append(
                EditorClassificationResult(
                    score_unmediated=score_u.item(),
                    score_mediated=score_m.item(),
                    sample={
                        key: sample[key]
                        for key in data.ContextMediationSample.__required_keys__
                    },
                )
            )

        return EditorClassifyRun(results)


class LinearEditor(Editor):
    """A simple linear model, optionally with a rank constraint."""

    def __init__(
        self,
        *,
        mt: models.ModelAndTokenizer,
        layer: int,
        rank: Optional[int] = None,
        use_entity: bool = False,
        use_attribute: bool = True,
        **kwargs: Any,
    ):
        """Initialize the editor.

        Args:
            rank: Rank constraint on the linear transformation. Defaults to None.
            use_entity: Use entity rep as input.
            use_attribute: Use attribute rep as input.

        """
        if not use_entity and not use_attribute:
            raise ValueError("must set >= 1 of `use_entity` or `use_attribute`")

        super().__init__(mt=mt, layer=layer, **kwargs)

        self.rank = rank
        self.use_entity = use_entity
        self.use_attribute = use_attribute

        hidden_size = models.determine_hidden_size(mt)
        input_size = hidden_size * sum([use_entity, use_attribute])

        self.w: nn.Linear | nn.Sequential
        if rank is None:
            self.w = nn.Linear(input_size, hidden_size)
        else:
            self.w = nn.Sequential(
                nn.Linear(input_size, rank),
                nn.Linear(rank, hidden_size),
            )
        self.to_(mt)

    def __call__(
        self, *, attribute: torch.Tensor, entity: torch.Tensor
    ) -> torch.Tensor:
        """Compute the edit direction."""
        inputs = []
        if self.use_entity:
            inputs.append(entity)
        if self.use_attribute:
            inputs.append(attribute)
        assert len(inputs) > 0

        return self.w(torch.cat(inputs, dim=-1))


class BiaffineEditor(Editor):
    """Biaffine model that takes entity to edit rep and attribute rep as input."""

    def __init__(self, *, mt: models.ModelAndTokenizer, layer: int, **kwargs: Any):
        """Initialize the editor."""
        super().__init__(mt=mt, layer=layer, **kwargs)
        hidden_size = models.determine_hidden_size(mt)
        self.w_entity = nn.Linear(hidden_size, hidden_size)
        self.w_attribute = nn.Linear(hidden_size, hidden_size)
        self.to_(mt)

    def forward(self, *, entity: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
        """Compute the edit direction."""
        return self.w_entity(entity) + self.w_attribute(attribute)


class MlpEditor(Editor):
    """Two-layer MLP editor on entity rep and attribute rep."""

    def __init__(
        self,
        *,
        mt: models.ModelAndTokenizer,
        layer: int,
        use_entity: bool = False,
        use_attribute: bool = True,
        **kwargs: Any,
    ):
        """Initialize the editor."""
        if not use_entity and not use_attribute:
            raise ValueError("must set >= 1 of `use_entity` or `use_attribute`")

        super().__init__(mt=mt, layer=layer, **kwargs)

        self.use_entity = use_entity
        self.use_attribute = use_attribute

        hidden_size = models.determine_hidden_size(mt)
        input_size = hidden_size * sum([use_entity, use_attribute])

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.to_(mt)

    def forward(self, *, entity: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
        """Compute the edit direction."""
        inputs = []
        if self.use_entity:
            inputs.append(entity)
        if self.use_attribute:
            inputs.append(attribute)
        return self.mlp(torch.cat(inputs, dim=-1))


class RandomEditor(Editor):
    """An editor that just picks a random edit direction."""

    def __init__(
        self,
        *,
        mt: models.ModelAndTokenizer,
        layer: int,
        mean: torch.Tensor | None = None,
        variance: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the editor."""
        super().__init__(mt=mt, layer=layer, **kwargs)

        hidden_size = models.determine_hidden_size(mt)
        device = models.determine_device(mt)

        if mean is None:
            mean = torch.zeros(hidden_size)
        if variance is None:
            variance = torch.ones(hidden_size, hidden_size)

        self.mean: torch.Tensor
        self.register_buffer("mean", mean.to(device))

        self.variance: torch.Tensor
        self.register_buffer("variance", variance.to(device))

        self.to_(mt)

    def forward(self, *, attribute: torch.Tensor, **_: Any) -> torch.Tensor:
        """Select a random direction."""
        distribution = torch.distributions.MultivariateNormal(
            self.mean, covariance_matrix=torch.diag(self.variance)
        )
        size = cast(torch.Size, (len(attribute),))
        return distribution.sample(size)

    def fit(
        self,
        *,
        dataset: Dataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hold_out: float = DEFAULT_HOLD_OUT,
        device: Optional[Device] = None,
        **_: Any,
    ) -> EditorTrainingRun:
        """Estimate mean and variance of entity representations."""
        dataset = data.maybe_train_test_split(dataset, test_size=hold_out)

        self.mt.model.to(device)
        self.to(device)

        # We never need these columns, so exclude to avoid null collation.
        columns = data.column_names(
            dataset, exclude=["target_mediated", "target_unmediated", "source"]
        )

        rv = runningstats.Variance()
        with dataset.formatted_as("torch", columns=columns):
            loader = torch.utils.data.DataLoader(
                cast(torch.utils.data.Dataset, dataset["train"]), batch_size=batch_size
            )
            for batch in tqdm(loader, desc="estimate mean/cov"):
                if not precompute.has_entity_deltas(batch):
                    batch.update(
                        precompute.entity_deltas_from_batch(
                            self.mt, batch, layers=[self.layer], device=device
                        )
                    )
                rv.add(batch[f"prompt_in_context.entity.delta.{self.layer}"].float())
        self.mean[:] = rv.mean()
        self.variance[:] = rv.covariance()
        return EditorTrainingRun(dataset)


class ScalarMultipleEditor(Editor):
    """Editor that returns scalar multiple of attribute rep.

    Scalar is determined by linear function of entity and attribute.
    """

    def __init__(self, *, mt: models.ModelAndTokenizer, **kwargs: Any):
        """Initialize the editor."""
        super().__init__(mt=mt, **kwargs)
        hidden_size = models.determine_hidden_size(mt)
        self.scalar = nn.Linear(hidden_size, 1)

    def forward(self, *, entity: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
        """Return multiple of attribute."""
        scalar = self.scalar(attribute)
        return scalar * attribute


class IdentityEditor(Editor):
    """Editor that returns the attribute, unmodified."""

    def forward(self, *, entity: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
        """Return the attribute."""
        return attribute


class NullEditor(Editor):
    """Editor that just returns zero direction."""

    def forward(self, *, entity: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
        """Return the zero direction."""
        return torch.zeros_like(entity)


SUPPORTED_EDITORS = {
    "linear": LinearEditor,
    "biaffine": BiaffineEditor,
    "mlp": MlpEditor,
    "random": RandomEditor,
    "scalar": ScalarMultipleEditor,
    "identity": IdentityEditor,
    "null": NullEditor,
}


def load_editor(
    mt: models.ModelAndTokenizer,
    editor_type: str,
    layer: int,
    editors_dir: PathLike | None = None,
    device: Device | None = None,
) -> Editor | None:
    """Load editor of given type from the directory, assuming default options."""
    editor_factory = SUPPORTED_EDITORS[editor_type]
    editor = editor_factory(mt=mt, layer=layer)
    editor.to(device)

    if editor_type not in {"null", "identity"}:
        if editors_dir is None:
            logger.warning("editors_dir not specified for non-identity editor")
            return None

        weights_file = Path(editors_dir) / editor_type / str(layer) / "weights.pth"
        if not weights_file.exists():
            logger.warning(f"weights expected at {weights_file} but not found")
            return None

        logger.info(f"loading editor weights from {weights_file}")
        state_dict = torch.load(weights_file, map_location=device)
        editor.load_state_dict(state_dict)

    return editor


def save_editor(editor: Editor, editors_dir: PathLike) -> Path:
    """Save the editor in a properly indexed directory.

    Does NOT index on the underlying model or the dataset on which the editor
    was trained on, so callers should make sure `editors_dir` is unique to the
    model + dataset combo.

    Args:
        editor: The editor to save.
        editors_dir: The dir to save to.

    Returns:
        Path to saved weights file.

    """
    editors_dir = Path(editors_dir)
    editor_type = {et: name for name, et in SUPPORTED_EDITORS.items()}[type(editor)]
    weights_file = editors_dir / editor_type / str(editor.layer) / "weights.pth"
    weights_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(editor.state_dict(), weights_file)
    logger.info(f"saved editor to {weights_file}")
    return weights_file


def list_saved_editors(editors_dir: PathLike) -> DefaultDict[str, list[int]]:
    """Return configs of all editors saved in directory.

    Assumes the file structure laid out by `save_editors`. Return value is a mapping
    from editor type to layers for which there is a saved editor.
    """
    editors_dir = Path(editors_dir)

    editor_type_dirs = [d for d in editors_dir.iterdir() if d.is_dir()]

    editor_configs = defaultdict(list)
    for editor_type_dir in editor_type_dirs:
        editor_configs[editor_type_dir.name] = sorted(
            [
                int(layer_dir.name)
                for layer_dir in editor_type_dir.iterdir()
                if layer_dir.is_dir()
            ]
        )

    if not editor_configs:
        logger.warning(f"no editors found in {editors_dir}")

    return editor_configs


def add_editor_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for scripts that require loading a pretrained editor.

    Added args include:
        --editor-type (-t): Editor type to load.
        --editors-dir (-d): Directory containing editors. Expected to have
            layout like: <editors_dir> / <editor_type> / <layer> / weights.pth
        --layers (-l): Usually we want to play with editing at multiple layers.
            This arg specifies which layers to try editors for.

    """
    parser.add_argument(
        "--editor-type",
        "-t",
        choices=SUPPORTED_EDITORS,
        default="linear",
        help="editor type, inferred by default",
    )
    parser.add_argument(
        "--editors-dir", "-e", type=Path, help="path to editor experiment"
    )
    parser.add_argument(
        "--layers", "-l", nargs="+", type=int, help="layers to apply remedi at"
    )
