# Hessian Spectrum Analysis Along L2-Regularized Model Interpolation

This repository contains the code used to investigate the loss landscape of neural networks through Hessian eigenspectrum analysis during a regularized transition between two trained models.

## Overview

The code performs a sequence of fine-tuning experiments on the MNIST dataset using a fully connected neural network. Starting from a pretrained model, the network is optimized under an L2 penalty that constrains the parameters toward a second reference model.

For each regularization strength:

1. The model is trained using cross-entropy loss plus an L2 penalty.
2. Training and test metrics are recorded.
3. The resulting model parameters are saved.
4. The Hessian matrix of the training loss is probed through Hessian-vector products.
5. The largest Hessian eigenvalues are computed using the Lanczos algorithm (`scipy.sparse.linalg.eigsh`).

This procedure enables the study of how curvature properties of the loss landscape evolve as the model moves between two minima.

---

## Neural Network Architecture

```text
Input (28×28)
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

The output layer is used together with the cross-entropy loss function.

---

## Methodology

Let:

- θ denote the current network parameters.
- θ_ref denote the parameters of a reference model.

Training minimizes

L(θ) = L_CE(θ) + λ ||θ − θ_ref||²

where:

- L_CE is the average cross-entropy loss on MNIST.
- λ is the regularization strength.

The regularization coefficient is varied according to

```python
L2_strengths = [i * 1e-8 for i in range(501)]
```

corresponding to

```text
0 ≤ λ ≤ 5 × 10⁻⁶
```

For each value of λ, a new model is obtained and analyzed.

---

## Hessian Computation

The Hessian is never constructed explicitly.

Instead, Hessian-vector products are computed through automatic differentiation:

1. Compute the gradient of the loss.
2. Form the gradient-vector inner product.
3. Differentiate once more to obtain Hv.

The resulting linear operator is supplied to

```python
scipy.sparse.linalg.eigsh
```

which computes the largest Hessian eigenvalues using the Lanczos method.

The code extracts

```python
k = 100
```

largest-magnitude eigenvalues for every trained model.

---

## Requirements

Install the required packages with:

```bash
pip install torch torchvision numpy scipy
```

Dependencies:

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- SciPy

---

## Required Input Files

Before execution, the following pretrained models must be available:

```text
model_weights-3-0.00000.pth
model_weights-4-0.00000.pth
```

These files correspond to:

- The initial model used as the starting point of optimization.
- The reference model used in the L2 regularization term.

---

## Running

Execute:

```bash
python code.py
```

The script automatically:

1. Loads the MNIST dataset.
2. Loads the pretrained networks.
3. Sweeps over all regularization strengths.
4. Trains the network.
5. Computes Hessian eigenvalues.
6. Saves metrics and model checkpoints.

---

## Output Files

### Training Metrics

```text
Model-0-[3_to_4].csv
```

Columns:

```text
L2_strength
Train_Loss
Train_Reg_Loss
Test_Loss
Test_Accuracy
```

### Model Checkpoints

```text
model_weights-0-<L2_strength>-[3_to_4].pth
```

Saved after each regularization strength.

### Hessian Eigenvalues

```text
Hessian-eigenvalues-0-<L2_strength>-[3_to_4].csv
```

Contains the 100 largest Hessian eigenvalues.

---

## Reproducibility

Random seeds are fixed through

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```

and deterministic CUDA behavior is enabled whenever possible.

---

## Scientific Purpose

This code was developed to study:

- Loss landscape geometry of neural networks.
- Curvature evolution between distinct minima.
- Hessian spectral properties.
- Connections between optimization trajectories and sharpness.
- Effects of parameter-space regularization toward reference solutions.

The implementation accompanies the analyses reported in the associated publication.

---

## Citation

If you use this code in academic work, please cite the associated publication.
