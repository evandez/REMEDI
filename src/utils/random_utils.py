"""Helpful functions for dealing with randomness."""
import random

import numpy
import torch


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
