# Neural Network Regularization, Mode Connectivity & Ablation Experiments

This repository contains experimental data tracking the effects of $L_2$ regularization (weight decay) on neural network training dynamics, test performance, weight-space topology, and structural connectivity. 

The dataset is divided across standard optimization sweeps, architectural/dataset ablations, and path-connectivity landscapes.

---

## 📐 Fundamental Definitions

Throughout all experiments and datasets:
* **Distance Metric ($r$):** Universally defined as the **distance in the parameter space of weights and biases**.
* **Origin Reference ($r_0$):** Represents the baseline model configuration or **distance to the origin** (initialization state).

---

## 📂 Repository Structure & File Descriptions

### 1. Standard Learning Rate & Regularization Sweeps (`lr=*.csv`)
These files monitor core performance metrics (Train/Test Loss, Accuracy, and the parameter-space distance $r_0$ to the origin) across a broad gradient of $L_2$ penalties on the base dataset (MNIST). 

> 💡 **Note on Random Seeds:** File names ending in `_1`, `_2`, or `_3` represent identical experimental setups run with different **random seeds**. Multiple seeds are provided to account for stochastic variations in network weight initialization, dataset shuffling, and batching—ensuring the observed regularization trends are statistically reproducible.

* `lr=0.015_1.csv` & `lr=0.015_2.csv`: Independent seeded runs evaluating a higher learning rate ($\eta = 0.015$).
* `lr=0.0015_1.csv`, `lr=0.0015_2.csv`, & `lr=0.0015_3.csv`: Independent seeded runs evaluating a lower learning rate ($\eta = 0.0015$).
* `lr=0.0015_1_split_accuracies.csv`: An extended breakdown of the first $\eta = 0.0015$ run, tracking test accuracy across individual categories (`Class_0_Accuracy` through `Class_9_Accuracy`) to inspect regularized forgetting or class-specific sensitivity.

### 2. Dataset & Architectural Ablations
These files serve as controls or comparison baselines to verify if regularized training and distance behaviors generalize across different architectures or datasets at a base learning rate of $\eta = 0.0015$.

* `lr=0.0015_with_Fashion_MNST.csv`: Replicates the weight decay sweep using the **Fashion-MNIST** dataset instead of standard digits, validating dataset-invariant regularization phase transitions.
* `lr=0.0015_with_ReLU.csv`: Replicates the sweep using a standard **ReLU activation function** in the network architecture to isolate how different activation behaviors affect weight norm constraints and stability boundaries.

### 3. Path Connectivity & Optimization Paths
These files track loss landscapes, barrier energies, or structural configurations between different checkpoints or trajectories in the weight space.

* `connect_0_to_1.csv` to `connect_0_to_4.csv`: Tracks continuous connectivity metrics and parameter-space distances mapping from the baseline origin minimum ($r_0$) to four **different target minima** ($r_1, r_2, r_3, r_4$).
* `lr=0.0015_path_finder_T4.csv`: A specialized optimization trajectory tracking distance metrics relative to target endpoint $r_4$, where **$T_4$ represents the fourth discontinuity in the pathway**. This is used for evaluating path-finding efficiency or geometry under varying degrees of weight decay.

---

## 📊 Key Data Schema

Every CSV file contains an individual sweep per row. 

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `L2_strength` | Float | The coefficient applied to the $L_2$ regularization penalty term ($\lambda$). |
| `Train_Loss` | Float | Raw objective/cross-entropy loss evaluated on the training dataset. |
| `Train_Reg_Loss` | Float | Total training loss including the $L_2$ penalty: $\text{Loss} + \lambda \sum w^2$. |
| `Test_Loss` | Float | Evaluation loss on the unseen test dataset. |
| `Test_Accuracy` | Float | Global model classification accuracy percentage on the test set. |
| `r0` or `r0-rX` | Float | Distance $r$ in the parameter space of weights and biases relative to the origin $r_0$. |
| `r4` | Float | *(Only in path_finder_T4)* Distance $r$ in the parameter space of weights and biases tracking proximity to the fourth discontinuity checkpoint $r_4$. |
| `Class_X_Accuracy` | Float | *(Only in split_accuracies)* Performance percentage for a single specific class label ($0$ to $9$). |
