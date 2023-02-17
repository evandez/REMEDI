"""Invoke tasks for running experiments from the paper."""
from experiments import eval, sweep, train

from invoke import Collection

ns = Collection("x")
for module in (eval, train, sweep):
    ns.add_collection(Collection.from_module(module))
