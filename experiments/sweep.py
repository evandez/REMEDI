"""Sweep over layers, determining best layer to apply REMEDI."""
from experiments import utils
from experiments.aliases import *

from invoke import Collection, task


def _sweep_gen(c, model, dataset, device=None):
    editors_dir = utils.require_editors_dir(model, dataset)
    name = utils.experiment_name("sweep_gen", dataset=dataset, model=model)
    script = "eval_fact_gen" if dataset == CF else "eval_bias_gen"
    cmd = f"python -m scripts.{script} --small -m {model} -n {name} -e {editors_dir}"
    if dataset == BB:
        cmd += " --decontextualized"
    cmd = utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


@task
def sweep_gen_cf(c, model=DEFAULT_MODEL, device=None):
    """Sweep for best layer to apply REMEDI at in counterfact."""
    _sweep_gen(c, model, CF, device=device)


@task
def sweep_gen_bb(c, model=DEFAULT_MODEL, device=None):
    """Sweep for best layer to apply REMEDI at in biosbias."""
    _sweep_gen(c, model, BB, device=device)


@task
def sweep_ent_mc(c, model=DEFAULT_MODEL, device=None):
    """Sweep for best layer to apply REMEDI at in McRae."""
    editors_dir = utils.require_editors_dir(model, MC)
    name = utils.experiment_name("sweep_ent", dataset=MC, model=model)
    cmd = f"python -m scripts.eval_entailment --small -m {model} -n {name} -e {editors_dir}"
    cmd = utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


def _sweep_cls(c, model, dataset, device=None):
    editors_dir = utils.require_editors_dir(model, dataset)
    name = utils.experiment_name("sweep_cls", dataset=dataset, model=model)
    script = "eval_fact_cls" if dataset == CF else "eval_bias_cls"
    layer = REMEDI_EDITOR_LAYER[model][dataset]
    cmd = f"python -m scripts.{script} --small -m {model} -n {name} -e {editors_dir} -l {layer}"
    cmd = utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


@task
def sweep_cls_cf(c, model=DEFAULT_MODEL, device=None):
    """Sweep for best layer to take entity from during classification in counterfact."""
    _sweep_cls(c, model, CF, device=device)


@task
def sweep_cls_bb(c, model=DEFAULT_MODEL, device=None):
    """Sweep for best layer to take entity from during classification in biosbias."""
    _sweep_cls(c, model, BB, device=device)


ns = Collection()

ns_gen = Collection("gen")
ns_gen.add_task(sweep_gen_cf, CF)
ns_gen.add_task(sweep_gen_bb, BB)
ns.add_collection(ns_gen)

ns_cls = Collection("cls")
ns_cls.add_task(sweep_cls_cf, CF)
ns_cls.add_task(sweep_cls_bb, BB)
ns.add_collection(ns_cls)

ns_ent = Collection("ent")
ns_ent.add_task(sweep_ent_mc, MC)
ns.add_collection(ns_ent)
