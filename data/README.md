# Neural Network Regularization, Mode Connectivity & Ablation Experiments

This repository contains experimental data tracking the effects of $L_2$ regularization (weight decay) on neural network training dynamics, test performance, weight-space topology, and structural connectivity. 

The dataset is divided across standard optimization sweeps, architectural/dataset ablations, and path-connectivity landscapes.

---

## 📂 Repository Structure & File Descriptions

### 1. Standard Learning Rate & Regularization Sweeps (`lr=*.csv`)
These files monitor core performance metrics (Train/Test Loss, Accuracy, and distance to initialization $r_0$) across a broad gradient of $L_2$ penalties on the base dataset (MNIST).

* `lr=0.015_1.csv` & `lr=0.015_2.csv`: Repeats evaluating a higher learning rate ($\eta = 0.015$).
* `lr=0.0015_1.csv`, `lr=0.0015_2.csv`, & `lr=0.0015_3.csv`: Repeats evaluating a lower learning rate ($\eta = 0.0015$).
* `lr=0.0015_1_split_accuracies.csv`: An extended breakdown of the first $\eta = 0.0015$ run, tracking test accuracy across individual categories (`Class_0_Accuracy` through `Class_9_Accuracy`) to inspect regularized forgetting or class-specific sensitivity.

### 2. Dataset & Architectural Ablations
These files serve as controls or comparison baselines to verify if regularized training and distance behaviors generalize across different architectures or datasets at a base learning rate of $\eta = 0.0015$.

* `lr=0.0015_with_Fashion_MNST.csv`: Replicates the weight decay sweep using the **Fashion-MNIST** dataset instead of standard digits, validating dataset-invariant regularization phase transitions.
* `lr=0.0015_with_ReLU.csv`: Replicates the sweep using a standard **ReLU activation function** in the network architecture to isolate how different activation behaviors affect weight norm constraints and stability boundaries.

### 3. Path Connectivity & Optimization Paths
These files track loss landscapes, barrier energies, or structural configurations between different checkpoints or trajectories in the weight space.

* `connect_0_to_1.csv` to `connect_0_to_4.csv`: Tracks continuous weight-space distances or connectivity metrics mapping from a reference model baseline ($r_0$) to four distinct target training points ($r_1, r_2, r_3, r_4$).
* `lr=0.0015_path_finder_T4.csv`: A specialized optimization trajectory track tracking distance metrics relative to the fourth target endpoint ($r_4$), used for evaluating the path-finding efficiency or geometry under varying degrees of weight decay.

---

## 📊 Key Data Schema

Every CSV file contains an individual sweep per row. The columns map to the following primary attributes:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `L2_strength` | Float | The coefficient applied to the $L_2$ regularization penalty term ($\lambda$). |
| `Train_Loss` | Float | Raw objective/cross-entropy loss evaluated on the training dataset. |
| `Train_Reg_Loss` | Float | Total training loss including the $L_2$ penalty: $\text{Loss} + \lambda \sum w^2$. |
| `Test_Loss` | Float | Evaluation loss on the unseen test dataset. |
| `Test_Accuracy` | Float | Global model classification accuracy percentage on the test set. |
| `r0` | Float | Euclidean weight distance, energy, or path metric relative to the standard model origin/base state $r_0$. |
| `r4` | Float | *(Only in path_finder_T4)* Weight distance tracking proximity to the final target checkpoint $r_4$. |
| `Class_X_Accuracy` | Float | *(Only in split_accuracies)* Performance percentage for a single specific class label (0–9). |

---

## 🚀 Research Questions Explored

* **Generalization and Ablations:** Does the critical threshold where weight decay collapses optimization ($\text{Test\_Accuracy} \to \sim 10\%$) change significantly when swapping out data distributions (Fashion-MNIST) or activation constraints (ReLU)?
* **Subspace Connectivity:** How does increasing the $L_2$ penalty alter the length, energy barriers, or geometric layout of paths connecting distinct optima ($r_0 \to r_n$)?
* **Path Finding Dynamics:** Does heavy regularized training constrain the path trajectory closer to target subspaces ($r_4$) during optimization routines?

---

## 🛠 Quick Start Visualization

You can easily analyze any configuration run using standard scientific Python tools. For example, comparing the structural compression between the baseline and the ReLU variant:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load baseline and ReLU ablation files
df_base = pd.read_csv("lr=0.0015_1.csv")
df_relu = pd.read_csv("lr=0.0015_with_ReLU.csv")

plt.figure(figsize=(9, 5))
plt.plot(df_base["L2_strength"], df_base["r0"], label="Base Model Distance ($r_0$)", color="blue")
plt.plot(df_relu["L2_strength"], df_relu["r0"], label="ReLU Variant Distance ($r_0$)", color="orange", linestyle="--")

plt.xscale("log")
plt.xlabel("L2 Regularization Coefficient ($\lambda$)")
plt.ylabel("Weight Distance to Base State")
plt.title("Effect of Regularization on Weight Space Expansion")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.show()
