"""Sweep over layers, determining best layer to apply REMEDI."""
from experiments import utils
from experiments.aliases import *

from invoke import Collection, task

ns = Collection()


def _sweep_gen(c, model, dataset, device=None):
    editors_dir = utils.require_editors_dir(model, dataset)
    name = utils.experiment_name("sweep_fact_gen", dataset=dataset, model=model)
    script = "eval_fact_gen" if dataset == CF else "eval_bias_gen"
    cmd = f"python -m scripts.{script} --small -m {model} -n {name} -e {editors_dir}"
    utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


ns_gen = Collection("gen")
for dataset in (CF, BB):
    ns_dataset = Collection(dataset)
    for model in MODELS:

        @task
        def sweep_gen_task(c, device=None):
            f"""Sweep layers to pick best for REMEDI on {model}/{dataset}."""
            return _sweep_gen(c, model, CF, device=device)

        ns_dataset.add_task(sweep_gen_task, model)

    ns_gen.add_collection(ns_dataset)
ns.add_collection(ns_gen)


def _sweep_cls(c, model, dataset, device=None):
    editors_dir = utils.require_editors_dir(model, dataset)
    name = utils.experiment_name("sweep_fact_cls", dataset=dataset, model=model)
    script = "eval_fact_cls" if dataset == CF else "eval_bias_cls"
    layer = REMEDI_EDITOR_LAYER[model][dataset]
    cmd = f"python -m scripts.{script} --small -m {model} -n {name} -e {editors_dir} -l {layer}"
    utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


ns_cls = Collection("cls")
for dataset in (CF, BB):
    ns_dataset = Collection(dataset)
    for model in MODELS:

        @task
        def sweep_cls_task(c, device=None):
            f"""Sweep layers to pick best to take entity from for {model}/{dataset}."""
            _sweep_cls(c, model, dataset, device=device)

        ns_dataset.add_task(sweep_cls_task, model)
    ns_cls.add_collection(ns_dataset)
ns.add_collection(ns_cls)
