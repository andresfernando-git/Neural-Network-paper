# Neural Network Error Landscape Analysis

## Overview

This repository contains the code used to investigate the geometry of neural network error landscapes through regularization-induced phase transitions, Hessian eigenspectrum analysis, mode connectivity experiments, and convolutional-network studies. The implementations accompany the analyses presented in the associated publication and provide the tools used to generate all reported figures and datasets.

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

 value of λ, a new model is obtained and analyzed.

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
- Matplotlib
  
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

## Additional Notebooks

The repository also includes two Jupyter notebooks that complement the main Hessian analysis workflow.

### `make_all_figures.ipynb`

This notebook reproduces all figures presented in the paper directly from the processed CSV datasets contained in the data directory. It generates:

* MNIST phase-transition trajectories.
* Error and regularized-loss curves.
* Per-class accuracy evolution.
* Hessian eigenspectra.
* Mode-connectivity analyses.
* Generalization experiments on alternative datasets and architectures.

The notebook is intended as a visualization and figure-generation tool and does not require retraining any neural networks. It operates directly on the provided experimental data files.

### `CNN_experiments_error_beta.ipynb`

This notebook contains the implementation of the convolutional neural network experiments reported in the paper.

The experiment performs a regularization sweep on a CNN trained on a synthetic dataset composed of three noisy geometric patterns:

* Horizontal stripes
* Vertical stripes
* Checkerboard patterns

The network consists of convolutional layers followed by ReLU activations and pooling operations.  value of the L2 regularization strength, the notebook records:

* Training loss
* Regularized training loss
* Test loss
* Test accuracy
* Distance in parameter space from the origin

The resulting output is stored in:

```text
lr=0.01_with_CNNs_artifical_data.csv
```

This experiment serves as an architectural generalization test, investigating whether the regularization-induced phase transitions observed in fully connected neural networks persist in convolutional architectures trained on fundamentally different data.

---

## Repository Contents

```text
.
├── code.py
├── make_all_figures.ipynb
├── CNN_experiments_error_beta.ipynb
└── README.md
```

- `code.py`: Hessian eigenspectrum analysis and regularized model interpolation.
- `make_all_figures.ipynb`: Reproduces all figures presented in the paper.
- `CNN_experiments_error_beta.ipynb`: Convolutional-network experiments on synthetic pattern datasets.

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

- Regularization-induced phase transitions in neural networks.
- Loss landscape geometry and hierarchical basin structure.
- Curvature evolution between distinct minima.
- Hessian spectral properties and saddle-point formation.
- Mode connectivity between independently trained solutions.
- Effects of parameter-space regularization toward reference solutions.
- Generalization of landscape phenomena across datasets, activation functions, and architectures.

The implementation accompanies the analyses reported in the associated publication.

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{ersoy2026phasetransitions,
  title={Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks},
  author={Ersoy, Ibrahim Talha and Cardozo Licha, Andrés Fernando and Wiesner, Karoline},
  year={2026}
}
```
