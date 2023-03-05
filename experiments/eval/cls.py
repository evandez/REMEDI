"""Evaluate REMEDI as a classifier."""
from experiments import utils
from experiments.aliases import *

from invoke import task


def _eval_cls(c, model, dataset, device=None):
    name = utils.experiment_name("eval_cls", model=model, dataset=dataset)
    editor_layer = REMEDI_EDITOR_LAYER[model][dataset]
    entity_layer = REMEDI_ENTITY_CLS_LAYER[model][dataset]
    script = "eval_fact_cls" if dataset == CF else "eval_bias_cls"
    cmd = (
        f"python -m scripts.{script} -n {name} "
        f"-l {editor_layer} --entity-layer {entity_layer}"
    )
    cmd = utils.maybe_set_device(cmd, device=device)

    # First: run experiments on real editor.
    editors_dir = utils.require_editors_dir(model=model, dataset=dataset)
    c.run(f"{cmd} -m {model} -e {editors_dir}")

    # Next: Run REMEDI (I).
    c.run(f"{cmd} -m {model} -t identity")

    # Next: Run control task.
    c.run(f"{cmd} -m {model} -e {editors_dir} --control-task")

    # Finally: Control model.
    control_model = f"{model}_random"
    control_editors_dir = utils.require_editors_dir(
        model=control_model, dataset=dataset
    )
    c.run(f"{cmd} -m models/{control_model} -e {control_editors_dir} --control-model")


@task(name=CF)
def eval_fact_cls(c, model=DEFAULT_MODEL, device=None):
    """Evaluate REMEDI as a classifier for factual knowledge."""
    _eval_cls(c, model, CF, device=device)


@task(name=BB)
def eval_bias_cls(c, model=DEFAULT_MODEL, device=None):
    """Evaluate REMEDI as a classifer for contextual knowledge."""
    _eval_cls(c, model, BB, device=device)
