# Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks

Official repository accompanying the paper:

**Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks**

**Authors:** Ibrahim Talha Ersoy, Andrés Fernando Cardozo Licha, Karoline Wiesner

---

## Overview

Deep neural networks exhibit several intriguing phenomena that remain poorly understood, including phase transitions during learning, the ubiquity of saddle points in high-dimensional loss landscapes, and mode connectivity between independently trained solutions.

In this work, we demonstrate that these phenomena share a common geometric origin. We show analytically that L2-regularization induces saddle points at concave boundaries of the error landscape, giving rise to phase transitions between distinct accuracy regimes. Building on this insight, we introduce the **Pathfinder algorithm**, a simple and computationally efficient method for exploring neural network error landscapes.

Through experiments on MNIST, Fashion-MNIST, and synthetic datasets, we uncover a hierarchical organization of accuracy basins, reveal their associated saddle points, and identify low-error paths connecting independently trained minima.

---

## Main Contributions

* Analytical explanation of regularization-induced phase transitions in deep neural networks.
* Identification of saddle points as the geometric mechanism underlying these transitions.
* Introduction of the **Pathfinder algorithm** for systematic exploration of error landscapes.
* Discovery of hierarchical accuracy basins in neural network parameter spaces.
* Demonstration of mode connectivity using Pathfinder trajectories.
* Validation across multiple datasets, activation functions, and architectures.

---

## Pathfinder Algorithm

The Pathfinder algorithm repurposes L2 regularization as a probe of landscape geometry. By gradually increasing the regularization strength and retraining the model to convergence, Pathfinder traces trajectories through parameter space and reveals:

* Phase transitions
* Saddle points
* Basin boundaries
* Mode connectivity
* Hierarchical landscape structure

Unlike many existing landscape exploration methods, Pathfinder relies only on standard training procedures and does not require an additional optimization loop.

---

## Repository Structure

```text
.
├── code for running the experiments/
├── data/
└── README.md
```

Detailed descriptions of the contents of each directory are provided in the corresponding folder-specific README files.

---

## Reproducibility

The repository contains all code required to reproduce the numerical experiments presented in the paper, including:

* MNIST experiments
* Fashion-MNIST experiments
* Synthetic CNN experiments
* Pathfinder trajectories
* Hessian spectrum analysis
* Mode connectivity studies
* Figure generation

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{ersoy2026phasetransitions,
  title={Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks},
  author={Ersoy, Ibrahim Talha and Cardozo Licha, Andrés Fernando and Wiesner, Karoline},
  year={2026}
}
```

---

## Contact

For questions regarding the manuscript, code, or reproducibility, please open an issue in this repository.

---

## License

This repository is provided for academic and research purposes.
