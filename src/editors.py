"""Editing models."""
import contextlib
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast, overload

from src import precompute
from src.utils import dataset_utils, model_utils, tokenizer_utils, training_utils
from src.utils.typing import (
    Dataset,
    Device,
    Model,
    ModelGenerateOutput,
    ModelOutput,
    Tokenizer,
)

import torch
import torch.utils.data
import transformers.modeling_outputs
from baukit import nethook, runningstats
from dataclasses_json import DataClassJsonMixin
from torch import nn, optim
from tqdm.auto import tqdm

DEFAULT_ALPHA = 1.0
DEFAULT_LAM_ADV = 1.0
DEFAULT_LAM_KL = 0.1
DEFAULT_N_TOP = 10
DEFAULT_N_GENERATE = 10
DEFAULT_BATCH_SIZE = 32
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
        alpha: Step size for applying the direction.

    """

    def __init__(
        self,
        *,
        model: Model,
        layer: int,
        directions: torch.Tensor,
        token_ranges: torch.Tensor,
        alpha: float = DEFAULT_ALPHA,
    ):
        """Initialize the context manager."""
        self.model = model
        self.layer = layer
        self.directions = directions
        self.token_ranges = token_ranges
        self.alpha = alpha
        self._trace = None

    def __enter__(self) -> Model:
        """Hook the model so the direction is applied."""
        [layer_path] = model_utils.determine_layer_paths(
            self.model, layers=[self.layer]
        )

        def edit_output(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            """Apply the directions."""
            if output[0].shape[1] == 1:
                # This condition is satisfied only when we're generating beyond the
                # prompt, in which case, we never want to edit the output.
                return output
            for bi, (i, j) in enumerate(self.token_ranges.tolist()):
                output[0][bi, i:j] = (
                    output[0][bi, i:j] + self.alpha * self.directions[bi]
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


class EditedModel(nn.Module):
    """Wrapper around an LM which knows how to automatically apply mediation.

    Importantly, this model assumes inputs look like `ContextMediationSample`s
    or batches of them. See `src/utils/dataset_utils.py` for more details.
    """

    def __init__(
        self,
        editor: "Editor",
        mt: Optional[model_utils.ModelAndTokenizer] = None,
        alpha: float = DEFAULT_ALPHA,
        device: Optional[Device] = None,
    ):
        """Wrap the model to be edited."""
        super().__init__()
        self.mt = mt if mt is not None else editor.mt
        self.editor = editor
        self.alpha = alpha
        self.device = device

    @overload
    def _run_edited_model(
        self,
        batch: dataset_utils.ContextMediationInput,
        generate: Literal[False] = ...,
        inputs: Optional[transformers.BatchEncoding] = ...,
        **kwargs: Any,
    ) -> ModelOutput:
        ...

    @overload
    def _run_edited_model(
        self,
        batch: dataset_utils.ContextMediationInput,
        generate: Literal[True],
        inputs: Optional[transformers.BatchEncoding] = ...,
        **kwargs: Any,
    ) -> ModelGenerateOutput:
        ...

    def _run_edited_model(
        self,
        batch: dataset_utils.ContextMediationInput,
        generate: bool = False,
        inputs: Optional[transformers.BatchEncoding] = None,
        **kwargs: Any,
    ) -> ModelOutput | ModelGenerateOutput:
        """Run the model on the inputs, editing its hidden reps in the process."""
        layer = self.editor.layer

        precomputed = {**batch}
        if not precompute.has_editor_inputs(precomputed):
            precomputed = precompute.editor_inputs_from_batch(
                self.mt,
                batch,
                layers=[layer],
                device=self.device,
                return_target_token_ids=False,
            )

        if self.mt.tokenizer.padding_side == "right":
            token_range_key = "token_range"
        else:
            token_range_key = "negative_token_range"

        entity_ij_key = f"prompt.entity.{token_range_key}"
        if self.editor.edit_last_entity_token:
            entity_ij_key += ".last"

        hiddens_entity_key = f"entity.entity.hiddens.{layer}."
        if self.editor.input_last_entity_token:
            hiddens_entity_key += "last"
        else:
            hiddens_entity_key += "average"

        entity_ij = precomputed[entity_ij_key]
        hiddens_entity = precomputed[hiddens_entity_key]
        hiddens_attr = precomputed[f"context.attribute.hiddens.{layer}"]

        # Make type checker happy and reformat.
        dtype = self.mt.model.config.torch_dtype
        hiddens_entity = cast(torch.Tensor, hiddens_entity).to(self.device, dtype)
        hiddens_attr = cast(torch.Tensor, hiddens_attr).to(self.device, dtype)
        entity_ij = cast(torch.Tensor, entity_ij)

        directions = self.editor(entity=hiddens_entity, attribute=hiddens_attr)

        if inputs is None:
            prompt = batch["prompt"]
            inputs, _ = precompute.inputs_from_batch(
                self.mt, prompt, device=self.device
            )

        with apply_direction(
            model=self.mt.model,
            layer=layer,
            directions=directions,
            token_ranges=entity_ij,
            alpha=self.alpha,
        ) as model:
            if generate:
                kwargs.setdefault("max_new_tokens", DEFAULT_N_GENERATE)
                kwargs.setdefault("pad_token_id", self.mt.tokenizer.eos_token_id)
                outputs = model.generate(**inputs, **kwargs)
            else:
                outputs = model(**inputs, **kwargs)

        return outputs

    def forward(
        self,
        batch: dataset_utils.ContextMediationInput,
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
        return self._run_edited_model(batch, inputs=inputs, generate=False, **kwargs)

    def generate(
        self,
        batch: dataset_utils.ContextMediationInput,
        inputs: Optional[transformers.BatchEncoding] = None,
        **kwargs: Any,
    ) -> ModelGenerateOutput:
        """Forwards to `mt.model.generate`, but still applies editor."""
        return self._run_edited_model(batch, inputs=inputs, generate=True, **kwargs)


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
        mt: Optional[model_utils.ModelAndTokenizer] = None,
        alpha: float = DEFAULT_ALPHA,
        device: Optional[Device] = None,
    ):
        """Initialize the context manager."""
        self.mt = mt
        self.editor = editor
        self.alpha = alpha
        self.device = device
        self._hooked: EditedModel | None = None

    def __enter__(
        self,
    ) -> EditedModelAndTokenizer:
        """Wrap the model."""
        self._hooked = EditedModel(
            self.editor, mt=self.mt, alpha=self.alpha, device=self.device
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
    lam_adv: float | None = DEFAULT_LAM_ADV,
    lam_kl: float | None = DEFAULT_LAM_KL,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Apply the edit to the representat

    See `src.precompute.editor_inputs_from_dataset` for expected format of `batch`.
    """
    prompt = batch["prompt"]
    mediated_token_ids = batch["target_mediated.token_id"]
    unmediated_token_ids = batch["target_unmediated.token_id"]
    batch_size = mediated_token_ids.shape[0]

    inputs, _ = precompute.inputs_from_batch(editor.mt, prompt, device=device)
    with apply(editor, device=device) as mt_edit:
        outputs_edit = mt_edit.model(batch, inputs=inputs)
        logp_edit = torch.log_softmax(outputs_edit.logits, dim=-1)

    batch_idx = torch.arange(batch_size)
    last_idx = precompute.last_token_index_from_batch(inputs)

    # Compute simple loss: the probability of the target token post-edit.
    loss = -logp_edit[batch_idx, last_idx, mediated_token_ids].mean()

    # If requested, penalize probability mass on the unmediated token.
    if lam_adv is not None:
        logp_edit_unmediated = logp_edit[batch_idx, last_idx, unmediated_token_ids]
        p_edit_unmediated = torch.exp(logp_edit_unmediated)
        loss_adv = torch.log(1 - p_edit_unmediated).mean()
        loss -= lam_adv * loss_adv

    # If requested, include a KL loss term with the original token distribution.
    if lam_kl is not None:
        with torch.inference_mode():
            outputs_orig = editor.mt.model(**inputs)
        logp_orig = torch.log_softmax(outputs_orig.logits, dim=-1)
        logp_orig = logp_orig[batch_idx, last_idx]
        logp_edit = logp_edit[batch_idx, last_idx]
        loss += lam_kl * nn.functional.kl_div(
            logp_edit, logp_orig, reduction="batchmean", log_target=True
        )

    return loss


@dataclass(frozen=True)
class EditorTrainingRun:
    """Results from running `Editor.fit`."""

    dataset: Dataset


@dataclass(frozen=True)
class EditorEvaluationResult(DataClassJsonMixin):
    """Result for single sample from `Editor.evaluate`."""

    sample: dict[str, str]

    # Note: Need explicit types here so DataClassJsonMixin works.
    before_top_tokens: list[str]
    before_top_logps: list[float]
    before_generations: list[str]

    after_top_tokens: list[str]
    after_top_logps: list[float]
    after_generations: list[str]

    before_target_mediated_score: Optional[float] = None
    before_target_unmediated_score: Optional[float] = None

    after_target_mediated_score: Optional[float] = None
    after_target_unmediated_score: Optional[float] = None


@dataclass(frozen=True)
class EditorEvaluateRun(DataClassJsonMixin):
    """Wrapper around a list of individual evaluation results."""

    results: list[EditorEvaluationResult]


class Editor(nn.Module):
    """A simple linear editing model."""

    def __init__(
        self,
        *,
        mt: model_utils.ModelAndTokenizer,
        layer: int,
        input_last_entity_token: bool = False,
        edit_last_entity_token: bool = False,
    ):
        """Initialize the editor."""
        super().__init__()
        self.mt = mt
        self.layer = layer
        self.input_last_entity_token = input_last_entity_token
        self.edit_last_entity_token = edit_last_entity_token

    def to_(self, mt: model_utils.ModelAndTokenizer | None = None) -> None:
        """Make this editor's device/dtype match the underlying models."""
        mt = self.mt if mt is None else mt
        self.to(
            device=model_utils.determine_device(mt),
            dtype=model_utils.determine_dtype(mt),
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
        lam_adv: float | None = DEFAULT_LAM_ADV,
        lam_kl: float | None = DEFAULT_LAM_KL,
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
            lam_adv: Loss weight for adversarial log[1 - p(unmediated)] term.
            lam_kl: Loss weight for KL div on next token distribution for prompt.
            patience: Stop after val loss does not improve for this many epochs.
            device: Run editor and model on this device.

        """
        dataset = dataset_utils.maybe_train_test_split(dataset, test_size=hold_out)

        self.mt.model.to(device)
        self.to(device)

        self.mt.model.eval()
        for parameter in self.mt.model.parameters():
            parameter.requires_grad_(True)

        # NOTE(evandez): If we're using half precision, we have to increase the eps
        # used by Adam or we'll immediately get NaNs everywhere. It's because the
        # default eps gets rounded to 0 in half precision.
        optimizer_kwargs: dict = dict(lr=lr)
        if model_utils.determine_dtype(self.mt) is torch.float16:
            optimizer_kwargs["eps"] = 1e-4

        optimizer = optim.AdamW(self.parameters(), **optimizer_kwargs)
        stopper = training_utils.EarlyStopping(patience=patience)

        with dataset.formatted_as("torch"):
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
                        lam_adv=lam_adv,
                        lam_kl=lam_kl,
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
                            lam_adv=lam_adv,
                            lam_kl=lam_kl,
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
        n_generate: int = DEFAULT_N_GENERATE,
        alpha: float = DEFAULT_ALPHA,
        device: Optional[Device] = None,
    ) -> EditorEvaluateRun:
        """Evaluate the editor on a held out set.

        Args:
            dataset: Dataset to evaluate on.
            batch_size: Model batch size.
            n_top: Number of top words/probs to return.
            n_generate: Number of tokens to generate.
            alpha: Step size for applying edit directions.
            device: Send all data to this device. Defaults to None.

        Returns:
            The evaluation results, one per entry in the dataset.

        """
        self.mt.eval_()
        self.mt.to_(device)
        self.eval().to(device)
        include_target_probs = "target_mediated" in dataset.column_names

        results = []
        with dataset.formatted_as("torch"):
            loader = torch.utils.data.DataLoader(
                cast(torch.utils.data.Dataset, dataset), batch_size=batch_size
            )
            for batch in tqdm(loader, desc=f"evaluate editor (layer={self.layer})"):
                prompts = batch["prompt"]
                current_batch_size = len(prompts)

                # Need padding side to be left for batch generate.
                with model_utils.set_padding_side(self.mt, padding_side="left"):
                    inputs, _ = precompute.inputs_from_batch(
                        self.mt, prompts, device=device
                    )

                generate_kwargs = dict(
                    do_sample=False,
                    max_new_tokens=n_generate,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.mt.tokenizer.eos_token_id,
                )
                outputs_before = self.mt.model.generate(**inputs, **generate_kwargs)
                with apply(self, alpha=alpha, device=device) as edited_mt:
                    outputs_after = edited_mt.model.generate(
                        batch, inputs=inputs, **generate_kwargs
                    )

                batched_results: dict = {
                    "sample": [
                        {
                            key: batch[key][bi]
                            for key in dataset_utils.ContextMediationSample.__required_keys__
                        }
                        for bi in range(current_batch_size)
                    ]
                }
                for key, outputs in (
                    ("before", outputs_before),
                    ("after", outputs_after),
                ):
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
                    # TODO(evandez): Fix typing here, should be list of lists.
                    batched_results[f"{key}_generations"] = generations

                    if include_target_probs:
                        batch_indices = torch.arange(current_batch_size)
                        for target_key in ("mediated", "unmediated"):
                            target_id = batch[f"target_{target_key}.token_id"]
                            target_probs = first_token_logps[batch_indices, target_id]
                            target_prob_key = f"{key}_target_{target_key}_score"
                            batched_results[target_prob_key] = target_probs.tolist()

                # Flatten results.
                for bi in range(current_batch_size):
                    result_kwargs: dict
                    result_kwargs = {k: vs[bi] for k, vs in batched_results.items()}
                    result = EditorEvaluationResult(**result_kwargs)
                    results.append(result)

            return EditorEvaluateRun(results)


class RandomEditor(Editor):
    """An editor that just picks a random edit direction."""

    def __init__(
        self,
        *,
        mt: model_utils.ModelAndTokenizer,
        layer: int,
        mean: torch.Tensor | None = None,
        covariance: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the editor."""
        super().__init__(mt=mt, layer=layer, **kwargs)

        hidden_size = model_utils.determine_hidden_size(mt)
        device = model_utils.determine_device(mt)

        if mean is None:
            mean = torch.zeros(hidden_size)
        if covariance is None:
            covariance = torch.ones(hidden_size)

        self.mean: torch.Tensor
        self.register_buffer("mean", mean.to(device))

        self.covariance: torch.Tensor
        self.register_buffer("covariance", covariance.to(device))

        self.to_(mt)

    def forward(self, *, attribute: torch.Tensor, **_: Any) -> torch.Tensor:
        """Select a random direction."""
        distribution = torch.distributions.MultivariateNormal(
            self.mean, self.covariance
        )
        return distribution.sample((len(attribute),))

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
        dataset = dataset_utils.maybe_train_test_split(dataset, test_size=hold_out)

        self.mt.model.to(device)
        self.to(device)

        rc = runningstats.Covariance()
        with dataset.formatted_as("torch"):
            loader = torch.utils.data.DataLoader(
                cast(torch.utils.data.Dataset, dataset["train"]), batch_size=batch_size
            )
            for batch in tqdm(loader, desc="estimate mean/cov"):
                if not precompute.has_entity_deltas(batch):
                    batch = precompute.entity_deltas_from_batch(
                        self.mt, batch, layers=[self.layer], device=device
                    )
                rc.add(batch[f"prompt_in_context.delta.{self.layer}"])

        self.mean[:] = rc.mean()
        self.covariance[:] = rc.covariance()
        return EditorTrainingRun(dataset)


class LinearEditor(Editor):
    """A simple linear model, optionally with a rank constraint."""

    def __init__(
        self,
        *,
        mt: model_utils.ModelAndTokenizer,
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

        hidden_size = model_utils.determine_hidden_size(mt)
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

    def __init__(self, *, mt: model_utils.ModelAndTokenizer, layer: int, **kwargs: Any):
        """Initialize the editor."""
        super().__init__(mt=mt, layer=layer, **kwargs)
        hidden_size = model_utils.determine_hidden_size(mt)
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
        mt: model_utils.ModelAndTokenizer,
        layer: int,
        use_entity: bool = True,
        use_attribute: bool = True,
        **kwargs: Any,
    ):
        """Initialize the editor."""
        if not use_entity and not use_attribute:
            raise ValueError("must set >= 1 of `use_entity` or `use_attribute`")

        super().__init__(mt=mt, layer=layer, **kwargs)

        self.use_entity = use_entity
        self.use_attribute = use_attribute

        hidden_size = model_utils.determine_hidden_size(mt)
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


# TODO(evandez): Small fixes needed for this file:
# - Need a way to have evaluation results point back to original dataset.
# - Consistency in counterfact splits (set seed)
# - Show running average loss in progress bar, not batch loss.
# - This currently tokenizes the prompt twice, can we avoid?
# - Precompute does everything on GPU, even averaging.
# - Set hyperparameter defaults by editor type.
