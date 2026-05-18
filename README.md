# OT Voyage

This repo contains a small, self-contained implementation of discrete OT tools, with notebooks illustrating Sinkhorn’s algorithm and a few applications to MNIST.

## Contents

```text
.
├── notebooks/
│   ├── sinkhorn.ipynb   # Sinkhorn from scratch on toy examples
│   └── mnist.ipynb      # OT experiments on MNIST images
├── src/
│   ├── core.py          # grids, costs, histogram normalization, Sinkhorn solvers
│   ├── baselines.py     # exact OT baselines for small problems
│   └── utils.py         # MNIST helpers, barycenters, retrieval utilities
└── outputs/             # generated figures and cached arrays
```

## Installation

Create a fresh environment and install the main dependencies:

```bash
pip install numpy scipy matplotlib pot torchvision
```

## Notes

The project includes a local Sinkhorn implementation in `src/core.py`. MNIST experiments also use [POT](https://pythonot.github.io/).

Generated figures and cached cost matrices are written under `outputs/`.

Feel free to reach out if you have any input :)
