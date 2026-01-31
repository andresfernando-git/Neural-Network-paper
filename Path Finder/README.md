# Hessian Spectral Dynamics of Anchor-Regularized Loss Landscapes

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg)](https://arxiv.org/)

This repository contains the implementation and experimental framework for analyzing the second-order geometry of neural network loss surfaces under **Anchor-Regularization**. 

The study focuses on the spectral evolution of the Hessian matrix $\mathbf{H}$ as the optimization objective is constrained toward a non-zero reference state $\theta_{\text{ref}}$, rather than the origin.

---

## 1. Theoretical Framework

### Anchor-Regularized Objective
We define the effective loss function $L_{\text{eff}}$ as a combination of the empirical risk and a quadratic penalty relative to a fixed anchor $\theta_{\text{ref}}$:

$$L_{\text{eff}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f(x_i; \theta), y_i) + \lambda \|\theta - \theta_{\text{ref}}\|_2^2$$

Where:
* $\mathcal{L}$ is the Cross-Entropy loss.
* $\lambda$ is the regularization strength (the "pull" toward the anchor).
* $\theta_{\text{ref}}$ represents a critical point or a previously discovered local minimum.

### Second-Order Analysis
The curvature of the landscape is dictated by the Hessian matrix $\mathbf{H} = \nabla^2 L_{\text{eff}}(\theta)$. Because $\mathbf{H} \in \mathbb{R}^{P \times P}$ (where $P$ is the number of parameters), explicit computation is avoided in favor of the **Pearlmutter Trick**. 



We compute Hessian-vector products (HVP) using double automatic differentiation:
$$\text{HVP}(v) = \nabla_{\theta} \left( \left( \nabla_{\theta} L(\theta) \right)^T v \right)$$

This allows the **Lanczos algorithm** to efficiently extract the top $k=30$ eigenvalues ($\lambda_1 \geq \lambda_2 \dots \geq \lambda_{30}$) and their corresponding eigenvectors.



---

## 2. Technical Implementation

### System Architecture
The experiments utilize a Multi-Layer Perceptron (MLP) architecture designed for derivative stability:
* **Activation Functions**: Sigmoid (ensures $C^2$ continuity, providing a well-defined Hessian).
* **Optimizer**: Adam with a fixed learning rate of $0.0015$.
* **Data**: MNIST (784-128-128-10).

### Pipeline Execution
1. **Anchor Loading**: The script loads a reference `.pth` state dict to act as $\theta_{\text{ref}}$.
2. **Lambda Sweep**: Iterates through 501 points of $\lambda \in [0, 0.1]$.
3. **Training**: Performs optimization with early stopping based on training loss plateau.
4. **Spectral Extraction**: Computes the top 30 eigenvalues and eigenvectors at the converged point for each $\lambda$.

---

## 3. Data Taxonomy & Output Structure

Execution generates a structured hierarchy of results necessary for manifold mapping:

| Filename | Type | Description |
| :--- | :--- | :--- |
| `Model-{ID}.csv` | **Metrics** | $\lambda$, Train/Test Loss, and Test Accuracy. |
| `gradients-{ID}-{L2}.csv` | **First-Order** | Flattened gradient vectors $\nabla L$ for every layer. |
| `model_weights-{ID}-{L2}.pth` | **Parameters** | Serialized weights $\theta$ at convergence. |
| `Hessian-eigenvalues-{ID}-{L2}.csv` | **Spectral** | Top 30 eigenvalues (CSV column). |
| `Hessian-eigenvectors-{ID}-{L2}.csv` | **Geometric** | Directions of maximum curvature (eigenvectors). |

---

## 4. Installation & Reproducibility

### Environment Setup
```bash
# Recommended: Python 3.10+
pip install torch torchvision numpy scipy matplotlib
