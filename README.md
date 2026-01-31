# Spectral Analysis of Anchor-Regularized Loss Landscapes

This folder contains the results of a high-resolution sweep of the neural loss landscape. The experiments investigate how geometric curvature (Hessian spectrum) and model performance evolve when parameters are constrained toward a specific reference anchor $\theta_{ref}$.

---

## 1. Metrics Structure: `Model-{ID}.csv`

The primary training and validation telemetry is stored in `Model-0.csv`. Each row represents a converged state at a specific regularization increment.

### Column Descriptions

| Column | Variable | Definition |
| :--- | :--- | :--- |
| **`L2_strength`** | $\lambda$ | The regularization coefficient. It defines the "pull" strength toward the reference anchor $\theta_{ref}$. |
| **`Train_Loss`** | $L(\theta)$ | Raw Cross-Entropy loss on the training set. Measures empirical error without the penalty. |
| **`Train_Reg_Loss`** | $L_{eff}$ | The **Total Objective**: $L(\theta) + \lambda \|\theta - \theta_{ref}\|_2^2$. This is the value the optimizer actually minimized. |
| **`Test_Loss`** | $L_{test}$ | Cross-Entropy loss on the unseen MNIST test set. Used to monitor generalization. |
| **`Test_Accuracy`** | Acc % | The percentage of correct classifications on the test set. |

---

## 2. Weight Serialization: `.pth` Files

The files named `model_weights-[ID]-[L2].pth` are binary snapshots of the model's `state_dict`. 

### How to handle and load them
These files contain the weights and biases of the network. To reload a specific state for further analysis or visualization, use the following PyTorch pattern:

```python
import torch

# 1. Instantiate the architecture (must match the original Net class)
model = Net() 

# 2. Load the state dictionary
# Use map_location='cpu' if you are analyzing on a machine without a GPU
state_dict = torch.load('model_weights-0-0.02480.pth', map_location=torch.device('cpu'))

# 3. Load parameters into the model
model.load_state_dict(state_dict)

# 4. Set to evaluation mode
model.eval()
