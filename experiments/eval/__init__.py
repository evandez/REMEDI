"""Evaluation experiments."""
from experiments.eval import cls, gen

from invoke import Collection

ns = Collection()
ns.add_collection(Collection.from_module(gen))
ns.add_collection(Collection.from_module(cls))
