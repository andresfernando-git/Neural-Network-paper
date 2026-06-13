# Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks

Official repository accompanying the paper

**Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks**

**Ibrahim Talha Ersoy, Andrés Fernando Cardozo Licha, Karoline Wiesner**

---

## What if phase transitions, saddle points, and mode connectivity are all manifestations of the same geometric phenomenon?

Deep neural networks are trained on highly non-convex loss landscapes whose geometry remains poorly understood. This work demonstrates that three seemingly unrelated observations in deep learning

* Phase transitions during learning,
* Saddle points in the loss landscape,
* Mode connectivity between independently trained solutions,

all emerge from a common geometric origin.

We show analytically and numerically that L2 regularization transforms concave boundaries of the error landscape into saddle points of the loss landscape. As a consequence, regularization-induced phase transitions correspond to the crossing of these geometric boundaries.

---

## Pathfinder

To explore this geometry, we introduce the **Pathfinder algorithm**.

Instead of using L2 regularization merely as a tool to prevent overfitting, Pathfinder repurposes it as a probe of the loss landscape.

By gradually increasing a shifted L2 regularization term,

L(θ) = E(θ) + β ||θ − θref||²,

the algorithm traces controlled trajectories through parameter space and reveals:

* Hierarchical accuracy basins,
* Phase transitions,
* Saddle points,
* Flat connecting paths between minima,
* Feature acquisition and feature forgetting processes.

---

## Main Results

Using MNIST, Fashion-MNIST, and synthetic datasets, we show:

✓ Hierarchical phase transitions exist in multiple neural-network architectures.

✓ Phase transitions correspond to crossings of saddle points.

✓ Accuracy basins form a nested hierarchical structure in parameter space.

✓ Independently trained minima are connected through nearly flat low-error paths.

✓ The Pathfinder algorithm efficiently uncovers this geometry.

---

## Repository Contents

This repository contains:

* Source code used in the experiments.
* Data used to generate the figures presented in the paper.
* Hessian spectrum analyses.
* Pathfinder trajectory calculations.
* Supplementary materials supporting the results.

---

## Why This Matters

Understanding the geometry of neural-network error landscapes is fundamental for:

* Interpretability,
* Transfer learning,
* Continual learning,
* Model merging,
* Generalization theory,
* Statistical-physics approaches to deep learning.

The results presented here suggest that deep neural networks possess an intrinsic hierarchical organization that can be systematically explored through regularization.

---

## Paper

If you use this repository, please cite:

```bibtex
@article{ersoy2026phase,
  title={Phase Transitions Reveal Hierarchical Structure in Deep Neural Networks},
  author={Ersoy, Ibrahim Talha and Cardozo Licha, Andrés Fernando and Wiesner, Karoline},
  year={2026}
}
```

---

## Contact

For questions regarding the paper or the implementation, please contact the authors.
