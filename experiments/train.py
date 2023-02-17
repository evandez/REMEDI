"""Train REMEDI editors."""
from experiments import utils
from experiments.aliases import *

from invoke import task


def _train(c, model, dataset, device=None):
    name = utils.experiment_name("editors", model=model, dataset=dataset)
    cmd = f"python -m scripts.train_editors -m {model} -n {name} -d {dataset}"
    cmd = utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


def _maybe_random_model(c, model, random=False):
    if random:
        random_model = MODELS_DIR / f"{model}_random"
        if not random_model.exists():
            c.run(f"python -m scripts.random_init_model -m {model}")
        return random_model
    return model


@task(name=CF)
def train_cf(c, model=DEFAULT_MODEL, random=False, device=None):
    """Train REMEDI for factual editing."""
    model = _maybe_random_model(c, model, random=random)
    _train(c, model, CF, device=device)


@task(name=BB)
def train_bb(c, model=DEFAULT_MODEL, random=False, device=None):
    """Train REMEDI for context mediation."""
    model = _maybe_random_model(c, model, random=random)
    _train(c, model, BB, device=device)


@task(name=MC)
def train_mc(c, model=DEFAULT_MODEL, random=False, device=None):
    """Train REMEDI for entailment analysis."""
    model = _maybe_random_model(c, model, random=random)
    _train(c, model, MC, device=device)
