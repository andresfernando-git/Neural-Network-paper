---
language: en
license: mit
library_name: pytorch
tags:
- hessian-analysis
- spectral-geometry
- per-class-analysis
- mnist
- nature-ai
datasets:
- mnist
metrics:
- accuracy
- per-class-accuracy
- hessian-eigenvalues
- hessian-eigenvectors
---

# Multi-Scale Hessian Analysis and Class-Specific Generalization

This repository contains the advanced implementation for the second-order analysis of neural network loss landscapes. It extends standard spectral analysis by correlating Hessian curvature (eigenvalues and eigenvectors) with per-class generalization metrics under varying $L_2$ regularization regimes.

## 1. Research Scope

While typical optimization studies focus on aggregate loss, this framework investigates the **fine-grained geometry of the loss surface**. By extracting both eigenvalues and eigenvectors of the Hessian matrix, we can map the directions of maximal curvature to specific class-based performance degradation or improvement.

### Key Capabilities
* **High-Resolution Regularization Sweep**: Analysis across 451 increments of $\lambda \in [0, 0.09]$.
* **Spectral Decomposition**: Extraction of the top $k=30$ eigenvalues and their corresponding eigenvectors.
* **Granular Metrics**: Per-class accuracy tracking (0-9) to identify which semantic categories are most sensitive to curvature shifts.
* **Gradient Serialization**: Full-parameter gradient logging for every regularization step to facilitate trajectory analysis.

---

## 2. Mathematical Methodology

### Hessian-Vector Product (HVP)
To maintain computational efficiency for the 100k+ parameter space, we utilize the Pearlmutter Trick for the Hessian $\mathbf{H}$:

$$\mathbf{H}v = \nabla_{\theta} \left( \langle \nabla_{\theta} L(\theta), v \rangle \right)$$

This directional derivative is computed using PyTorch's `autograd` and integrated into a SciPy `LinearOperator` for Lanczos-based spectral decomposition.



### Class-Specific Accuracy
Generalization is monitored not just as a global mean, but as a vector $\mathbf{a} \in [0, 1]^{10}$, where each component $a_i$ represents:
$$a_i = \frac{\sum_{(x,y) \in \mathcal{D}_{test}} \mathbb{1}(\hat{y}=i \mid y=i)}{|\mathcal{D}_{test, i}|}$$

---

## 3. Technical Implementation

### Neural Network Architecture
The experiments utilize a Multi-Layer Perceptron (MLP) with Sigmoid activations to ensure a smooth loss landscape for second-order derivative stability.

| Layer | Type | Configuration |
| :--- | :--- | :--- |
| **Input** | Flatten | $784 \rightarrow 128$ |
| **FC1** | Linear + Sigmoid | $128 \rightarrow 128$ |
| **FC2** | Linear + Sigmoid | $128 \rightarrow 10$ |
| **Output** | Softmax | Cross-Entropy |

### Computational Requirements
* **Hardware**: NVIDIA GPU with CUDA support recommended (for `autograd` graph creation).
* **Memory**: High RAM required for large-scale gradient and eigenvector serialization.
* **Software**: Python 3.9+, PyTorch 2.0+, SciPy, NumPy.

---

## 4. Data Taxonomy and Artifacts

Execution of the main script populates the `test_data/` directory with the following assets:

| File Pattern | Description |
| :--- | :--- |
| `Model-[ID]_split_accuracy.csv` | Global metrics and per-class accuracies (0-9) per $\lambda$. |
| `Hessian-eigenvalues-[ID]-[L2].csv` | The top 30 eigenvalues of the Hessian matrix. |
| `Hessian-eigenvectors-[ID]-[L2].csv` | The corresponding eigenvectors (flattened parameter space). |
| `gradients-[ID]-[L2].csv` | Partial derivatives $\frac{\partial L}{\partial \theta}$ for all layers. |
| `model_weights-[ID]-[L2].pth` | Full model state dictionary for weight-space analysis. |

---

## 5. Usage

### 1. Environment Setup
```bash
pip install torch torchvision scipy numpy matplotlib
