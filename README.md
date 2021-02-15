# mogwai: Probabilistic Models of Protein Families

A library of tested models, metrics, and data loading for protein families. Implemented in PyTorch and PyTorch Lightning.

Under active development, feedback welcome.

## Getting Started

For now, we support cloning and installing in developer mode.

```bash
pip install -e .
```

You will also need to install `apex` (needed to use DDP and FusedLamb training):

```
source venv/bin/activate
git clone git@github.com:NVIDIA/apex.git
cd apex
```

Modify `setup.py`

Find:
```
if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
```
And replace it with
```
if (bare_metal_major != torch_binary_major):
```
To remove the minor version check. This will allow apex to install.

Install apex
```
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## Examples

* [Potts Model with Pseudolikelihood](https://github.com/nickbhat/mogwai/blob/main/examples/gremlin_train.ipynb)
