"""Evaluate REMEDI as an editor."""
from experiments import utils
from experiments.aliases import *

from invoke import Collection, task

ns = Collection()


def _eval_cf_gen(c, model, device=None):
    """Evaluate REMEDI on factual editing."""
    editors_dir = utils.require_editors_dir(model=model, dataset=CF)
    name = utils.experiment_name("eval_fact_gen", model=model)
    layer = REMEDI_EDITOR_LAYER[model][CF]
    cmd = f"python -m scripts.eval_fact_gen -n {name} -m {model}"
    utils.maybe_set_device(cmd, device=device)

    # Run editor eval followed by the baselines.
    c.run(f"{cmd} -e {editors_dir} -l {layer}")
    c.run(f"{cmd} --baseline prefix")
    c.run(f"{cmd} --baseline replace")


ns_cf = Collection(CF)
for model in MODELS:

    @task
    def eval_cf_gen_task(c, device=None):
        f"""Run factual generation eval for {model}."""
        _eval_cf_gen(c, model, device=device)

    ns_cf.add_task(eval_cf_gen_task, model)
ns.add_collection(ns_cf)


def _eval_bb_gen(c, model, device=None):
    """Evaluate REMEDI on context mediation correction."""
    editors_dir = utils.require_editors_dir(model=model, dataset=BB)
    name = utils.experiment_name("eval_bias_gen", model=model)
    layer = REMEDI_EDITOR_LAYER[model][BB]
    cmd = f"python -m scripts.eval_bias_gen -n {name} -m {model} -l {layer} -e {editors_dir}"
    utils.maybe_set_device(cmd, device=device)

    # Run contextual and decontextual case.
    c.run(cmd)
    c.run(f"{cmd} --decontextualized")


ns_bb = Collection(BB)
for model in MODELS:

    @task
    def eval_bb_gen_task(c, device=None):
        f"""Run context mediation generation eval for {model}."""
        _eval_bb_gen(c, model, device=device)

    ns_bb.add_task(eval_bb_gen_task, model)
ns.add_collection(ns_bb)
