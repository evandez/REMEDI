"""Evaluate REMEDI as a classifier."""
from experiments import utils
from experiments.aliases import *

from invoke import Collection, task


def _eval_fact_cls(c, model, device=None):
    editors_dir = utils.require_editors_dir(model=model, dataset=CF)
    name = utils.experiment_name("eval_fact_cls", model=model)
