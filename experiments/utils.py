"""Global aliases used in experiments."""
from experiments.aliases import *

from invoke import Exit


def experiment_name(key, dataset=None, model=None):
    """Generate experiment name."""
    name = f"{EX_PREFIX}{key}"
    if dataset is not None:
        name = f"{name}_{dataset}"
    if model is not None:
        name = f"{name}_{model}"
    return name


def experiment_results_dir(key, model, dataset):
    """Determine experiment results dir."""
    return RESULTS_DIR / experiment_name(key, dataset=dataset, model=model)


def require_editors_dir(model, dataset):
    """Assert dir containing trained editors exists and return it."""
    editors_dir = experiment_results_dir(EX_EDITORS, model=model, dataset=dataset)
    if not editors_dir.exists():
        raise Exit(message=f"editors not found at {editors_dir}", code=1)
    return editors_dir


def maybe_set_device(cmd, device=None):
    """Add --device arg to command if possible."""
    if device is not None:
        device = str(device).strip()
        if device.isdigit():
            device = f"cuda:{device}"
        cmd = cmd.rstrip() + f" --device {device}"
    return cmd
