# REMEDI: Editing Knowledge in Language Model Representations

[**Inspecting and Editing Knowledge Representations in Language Models**](https://arxiv.org/abs/2304.00740)<br>
[Evan Hernandez](https://evandez.com), [Belinda Z. Li](https://belindal.github.io), [Jacob Andreas](https://www.mit.edu/~jda/).<br>
<!-- TODO conference <br> -->

<hr>

This repository provides an implementation of the Representation Mediation (REMEDI) method for autoregressive transformer language models.

## Setup

All code is tested on `MacOS Ventura (>= 13.1)` and `Ubuntu 20.04` using `Python >= 3.10`. It uses a lot of newer Python features, so the Python version is a strict requirement.

To run the code, create a virtual environment with the tool of your choice, e.g. conda:

```bash
conda create --name remedi python=3.10
```

Then, after entering the environment, install the project dependencies:

```bash
python -m pip install invoke
invoke install
```

## Replicating Experiments

All experiments from the paper can be run through invoke. To see the full list, run:

```bash
invoke --list
```

Any task prefixed with an `x.` corresponds to an experiment. The invoke scripts have the hyperparameters from the paper baked into them. Most experiments support two flags: `--device` to specify the GPU, and `--model` to specify which LM to use (default: GPT-J).

### Training

The code supports training editors for most GPT variants: GPT2*, GPT-J, and GPT-NeoX (though Neo-X is too big with gradients for most single GPUs). In theory, the code also supports any autoregressive transformer LM, but this may need to slightly modify parts of `determine_hidden_size` and `determine_layers` inside the [models](remedi/models.py) module.

To run training with the default configuration, use invoke, e.g.:

```bash
invoke x.train.counterfact --device cuda
```

For more fine-grained control over the hyperparameters, run the training script directly, e.g.:

```bash
python -m scripts.train_editors \
    -n my_custom_editors \
    -m gptj \
    -d counterfact \
    -l 0 1 2 \
    --lam-kl 100 \
    --device cuda
```

The help strings for each command contain most of what you need to know.

### Evaluating

After training editors, you can evaluate them on any of the benchmarks considered in the paper. If you trained them via invoke, this is as simple as running another invoke command, typically one prefixed with `x.eval` e.g.:

```bash
invoke x.eval.gen.counterfact --device cuda
```

...which evaluate REMEDI on generation quality in counterfact.

Alterantively, as before, you can call the evaluation scripts directly.

```bash
python -m scripts.eval_fact_gen \
    -n my_custom_eval \
    -e results/my_custom_editors \
    -m gptj \
    -l 1 \
    --device cuda
```

## Contributing

While this library is not designed for industrial use (it's just a research project), we do believe research code should support reproducibility.  If you have issues running our code in the supported environment, please open an issue on this repository.

If you find ways to improve our code, you may also submit a pull request. Before doing so, please ensure that the code type checks, lints cleanly, and passes all unit tests. The following command should exit cleanly:

```bash
invoke presubmit
```

## How to Cite

```bibtex
@InProceedings{hernandez2023remedi,
  title     =   {Inspecting and Editing Knowledge Representations in Language Models},
  author    =   {Hernandez, Evan and Li, Belinda Z. and Andreas, Jacob},
  booktitle =   {Arxiv},
  year      =   {2023},
  url       =   {https://arxiv.org/abs/2304.00740}
}
```
