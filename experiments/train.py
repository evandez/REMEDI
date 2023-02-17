"""Train REMEDI editors."""
from experiments import utils
from experiments.aliases import *

from invoke import Collection, task

ns = Collection()


def _train(c, model, dataset, device=None):
    name = utils.experiment_name("editors", model=model, dataset=dataset)
    cmd = f"python -m scripts.train_editors -m {model} -n {name} -d {dataset}"
    cmd = utils.maybe_set_device(cmd, device=device)
    c.run(cmd)


for dataset in DATASETS:
    ns_dataset = Collection(dataset)
    for model in MODELS:

        @task
        def train_task(c, device=None):
            f"""Train REMEDI editors for {model}/{dataset}."""
            _train(c, model, dataset, device=device)

        ns_dataset.add_task(train_task, model)
    ns.add_collection(ns_dataset)
