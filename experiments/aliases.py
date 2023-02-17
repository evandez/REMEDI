import os
from pathlib import Path

# Assume all results are written to default results dir.
HERE = Path(__file__).parent
RESULTS_DIR = Path(os.getenv("CM_RESULTS_DIR", HERE / "results"))

# All models and datasets.
GPT2 = "gpt2"
GPT2_XL = "gpt2-xl"
GPTJ = "gptj"
MODELS = (GPT2, GPT2_XL, GPTJ)

CF = "counterfact"
BB = "biosbias"
MC = "mcrae"
DATASETS = (CF, BB, MC)

# Experiment keys.
EX_PREFIX = "post_icml_"  # Prepended to every experiment name.
EX_EDITORS = "editors"


# Layers to apply REMEDI edit to.
REMEDI_EDITOR_LAYER = {
    GPTJ: {
        CF: 1,
        BB: 11,
    },
}

# Layer to take entity from during classification.
REMEDI_ENTITY_CLS_LAYER = {
    GPTJ: {
        CF: 26,
        BB: 23,
    },
}
