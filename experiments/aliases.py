import os
from pathlib import Path

# Assume all results are written to default results dir.
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(os.getenv("CM_RESULTS_DIR", REPO_ROOT / "results"))
MODELS_DIR = Path(os.getenv("CM_MODELS_DIR", REPO_ROOT / "models"))

# All models and datasets.
GPT2 = "gpt2"
GPT2_XL = "gpt2-xl"
GPTJ = "gptj"

DEFAULT_MODEL = GPTJ

CF = "counterfact"
BB = "biosbias"
MC = "mcrae"

# Experiment keys.
EX_PREFIX = "emnlp_"  # Prepended to every experiment name.
EX_EDITORS = "editors"


# Layers to apply REMEDI edit to.
REMEDI_EDITOR_LAYER = {
    GPTJ: {
        CF: 1,
        BB: 12,
        MC: 12,
    },
    GPT2_XL: {
        CF: 0,
        BB: 4,
    },
    GPT2: {
        CF: 0,
        BB: 10,
    },
}

# Layer to take entity from during classification.
REMEDI_ENTITY_CLS_LAYER = {
    GPTJ: {
        CF: 26,
        BB: 15,
    },
    GPT2_XL: {
        CF: 34,
        BB: 37,
    },
    GPT2: {
        CF: 10,
        BB: 11,
    },
}
