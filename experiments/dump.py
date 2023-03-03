"""Dump directions."""
from experiments import utils
from experiments.aliases import *

from invoke import Collection, task


def _dump(c, model, dataset, device=None):
    name = utils.experiment_name("directions", model=model, dataset=dataset)
    editors_dir = utils.require_editors_dir(model=model, dataset=dataset)
    layer = REMEDI_EDITOR_LAYER[model][dataset]
    cmd = (
        f"python -m scripts.generate_directions -n {name} "
        f"-e {editors_dir} -d {dataset} -m {model} -l {layer}"
    )
    cmd = utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


@task
def dump_cf(c, model=DEFAULT_MODEL, device=None):
    """Generate dump of REMEDI directions for CounterFact."""
    _dump(c, model, CF, device=device)


@task
def dump_bb(c, model=DEFAULT_MODEL, device=None):
    """Generate dump of REMEDI directions for Bias in Bios."""
    _dump(c, model, BB, device=device)


@task
def dump_mc(c, model=DEFAULT_MODEL, device=None):
    """Generate dump of REMEDI directions for McRae."""
    _dump(c, model, MC, device=device)


@task
def dump_all(c, model=DEFAULT_MODEL, device=None):
    """Generate dump of REMEDI directions for all datasets."""
    _dump(c, model, CF, device=device)
    _dump(c, model, BB, device=device)
    _dump(c, model, MC, device=device)


ns = Collection()
ns.add_task(dump_cf, CF)
ns.add_task(dump_bb, BB)
ns.add_task(dump_mc, MC)
ns.add_task(dump_all, "all")
