"""Evaluation experiments."""
from experiments.eval import cls, ent, gen

from invoke import Collection

ns = Collection()
ns.add_collection(Collection.from_module(cls))
ns.add_collection(Collection.from_module(ent))
ns.add_collection(Collection.from_module(gen))
