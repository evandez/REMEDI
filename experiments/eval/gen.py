"""Evaluate REMEDI as an editor."""
from experiments import utils
from experiments.aliases import *

from invoke import task


@task(name=CF)
def eval_gen_cf(c, model=DEFAULT_MODEL, device=None):
    """Evaluate REMEDI on factual editing."""
    editors_dir = utils.require_editors_dir(model=model, dataset=CF)
    name = utils.experiment_name("eval_gen_counterfact", model=model)
    layer = REMEDI_EDITOR_LAYER[model][CF]
    cmd = f"python -m scripts.eval_fact_gen -n {name} -m {model}"
    cmd = utils.maybe_set_device(cmd, device=device)

    # Run editor eval followed by the baselines.
    c.run(f"{cmd} -e {editors_dir} -l {layer}")
    c.run(f"{cmd} --baseline prefix")
    c.run(f"{cmd} --baseline replace")


@task(name=BB)
def eval_gen_bb(c, model=DEFAULT_MODEL, device=None):
    """Evaluate REMEDI on context mediation correction."""
    editors_dir = utils.require_editors_dir(model=model, dataset=BB)
    name = utils.experiment_name("eval_gen_biosbias", model=model)
    layer = REMEDI_EDITOR_LAYER[model][BB]
    cmd = f"python -m scripts.eval_bias_gen -n {name} -m {model} -l {layer} -e {editors_dir}"
    cmd = utils.maybe_set_device(cmd, device=device)

    # Run contextual and decontextual case.
    c.run(cmd)
    c.run(f"{cmd} --decontextualized")
