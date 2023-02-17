"""Evaluation experiments."""
from experiments.eval import gen

from invoke import Collection

ns = Collection()
ns.add_collection(Collection.from_module(gen))
