# Loss Landscape Analysis Through Hessian Spectra and Model Interpolation

This repository contains the code and data accompanying the paper:

> [Paper Title]

The project investigates the geometry of neural network loss landscapes through Hessian eigenspectrum analysis during regularized transitions between different trained neural network solutions.

## Repository Structure

```text
.
├── Code for Running the Experiments/
│   ├── code.py
│   └── additional scripts used in the experiments
│
├── Data/
│   ├── r0/
│   └── r4/
│
└── README.md
```

### Code for Running the Experiments

This folder contains all scripts required to reproduce the experiments presented in the paper, including:

- Neural network training on MNIST.
- Regularized optimization between pretrained solutions.
- Hessian-vector product computation.
- Hessian eigenspectrum calculation using the Lanczos algorithm.
- Generation of performance metrics and model checkpoints.

### Data

The `Data` folder contains the experimental results analyzed in the paper.

The subdirectories

```text
r0/
r4/
```

contain the data associated with the two reference solutions used throughout the study. These files include the numerical results required to reproduce the figures, statistical analyses, and Hessian spectrum calculations reported in the manuscript.

## Scientific Motivation

The geometry of the loss landscape plays a fundamental role in neural network optimization and generalization. In particular, the Hessian matrix provides information about local curvature and can reveal structural differences between minima.

This repository investigates how the Hessian spectrum evolves when a neural network is constrained toward a second trained solution through parameter-space regularization.

## Experimental Setup

### Dataset

MNIST handwritten digit classification dataset.

### Neural Network Architecture

```text
Input (784)
    ↓
Linear(784 → 128)
    ↓
Sigmoid
    ↓
Linear(128 → 128)
    ↓
Sigmoid
    ↓
Linear(128 → 10)
```

### Regularized Optimization

The optimization objective is

L(θ) = LCE(θ) + λ ||θ − θref||²

where:

- LCE is the cross-entropy loss.
- θ denotes the current model parameters.
- θref denotes a reference trained solution.
- λ controls the regularization strength.

A sweep over λ generates a family of intermediate solutions connecting different regions of parameter space.

## Hessian Analysis

The Hessian matrix is accessed implicitly through Hessian-vector products computed with automatic differentiation.

The leading eigenvalues are obtained using the Lanczos method implemented in

```python
scipy.sparse.linalg.eigsh
```

allowing efficient analysis of the loss landscape curvature.

## Requirements

```bash
pip install torch torchvision numpy scipy
```

Dependencies:

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- SciPy

## Running the Experiments

Navigate to:

```text
Code for Running the Experiments/
```

and run:

```bash
python code.py
```

The script automatically:

1. Loads the MNIST dataset.
2. Loads the reference solutions.
3. Performs the regularized training procedure.
4. Computes performance metrics.
5. Calculates Hessian eigenvalues.
6. Saves the resulting data.

## Reproducibility

All experiments use fixed random seeds to ensure reproducibility whenever possible.

## Citation

If you use this repository in your research, please cite the associated publication.

```bibtex
@article{YOUR_CITATION,
  title={Paper Title},
  author={Authors},
  journal={Journal},
  year={Year}
}
```
