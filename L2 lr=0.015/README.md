---
language: en
license: mit
library_name: pytorch
tags:
- hessian-spectral-analysis
- loss-landscape-geometry
- second-order-optimization
- mnist
- nature-ai
datasets:
- mnist
metrics:
- test-accuracy
- hessian-eigenvalues
- spectral-outliers
---

# Spectral Geometry of Neural Loss Landscapes under Regularization

This repository contains the primary implementation for analyzing the second-order curvature of neural network loss landscapes. The framework explores the relationship between $L_2$ regularization strength, gradient flow, and the spectral density of the Hessian matrix.

## 1. Scientific Objective

The goal of this research is to characterize how the curvature of the loss surface—represented by the Hessian matrix $\mathbf{H}$—evolves as the optimization objective is constrained by varying degrees of weight decay. By extracting the top $k=30$ eigenvalues and their corresponding eigenvectors, we provide a high-resolution view of the "outlier" dimensions that dominate the optimization dynamics.

---

## 2. Mathematical Framework

### Hessian-Vector Product (HVP) Implementation
Directly computing the Hessian $\mathbf{H} \in \mathbb{R}^{P \times P}$ is computationally infeasible for modern neural networks. This implementation utilizes a matrix-free approach (the Pearlmutter Trick) to compute the product of the Hessian and an arbitrary vector $v$:

$$\mathbf{H}v = \nabla_{\theta} \left( \langle \nabla_{\theta} L(\theta), v \rangle \right)$$

This is achieved via double automatic differentiation in PyTorch, allowing the Lanczos algorithm (via `scipy.sparse.linalg.eigsh`) to iteratively converge on the largest eigenvalues (LM - Largest Magnitude).



### Regularized Optimization
The effective loss function $L_{\text{eff}}$ utilized during the sweep is:

$$L_{\text{eff}}(\theta) = L_{\text{CE}}(\theta) + \lambda \|\theta\|_2^2$$

where $L_{\text{CE}}$ is the Cross-Entropy loss and $\lambda$ represents the `L2_strength` ranging from $0$ to $0.09$.

---

## 3. System Architecture

### Model Specification
A 3-layer Multi-Layer Perceptron (MLP) designed for maximum derivative stability:

* **Activations**: Sigmoid (applied to hidden layers to ensure smooth second-order gradients).
* **Input**: $784$ (Flattened MNIST).
* **Hidden Layers**: Two layers of $128$ neurons.
* **Output**: $10$ (Linear logits for Cross-Entropy).

### Experimental Pipeline
1. **Deterministic Seeding**: Global seeds are set for `torch`, `cuda`, `numpy`, and `random`.
2. **Systematic Sweep**: 451 training iterations with linearly increasing $L_2$ penalties.
3. **Early Stopping**: Convergence is gated by a training loss patience of 5 epochs to ensure analysis is performed at local minima.
4. **Spectral Decomposition**: Post-training computation of the top 30 eigenvalues and eigenvectors.

---

## 4. Data Taxonomy

The execution populates a `test_data/` directory with the following artifacts for reproducibility:

| File Type | Naming Convention | Contents |
| :--- | :--- | :--- |
| **Metrics** | `Model-{ID}.csv` | $L_2$ strength, Train/Test Loss, Global Accuracy. |
| **Gradients** | `gradients-{ID}-{L2}.csv` | Full parameter-wise gradient vectors $\nabla L$. |
| **Eigenvalues** | `Hessian-eigenvalues-{ID}-{L2}.csv` | The top 30 eigenvalues ($\lambda_1, \dots, \lambda_{30}$). |
| **Eigenvectors** | `Hessian-eigenvectors-{ID}-{L2}.csv` | Flattened eigenvectors corresponding to $\mathbf{H}$. |
| **Weights** | `model_weights-{ID}-{L2}.pth` | Serialized PyTorch state dictionary. |

---

## 5. Usage and Installation

### Dependencies
* Python 3.10+
* PyTorch 2.x
* SciPy 1.10+
* NumPy

### Execution
```bash
# Clone the repository
git clone [https://github.com/your-username/hessian-spectral-analysis.git](https://github.com/your-username/hessian-spectral-analysis.git)
cd hessian-spectral-analysis

# Run the complete sweep and spectral analysis
python main.py
