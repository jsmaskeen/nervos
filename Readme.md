# Nervos

## A Spiking Neural Network Simulation Framework

**Nervos** is a flexible and customizable tool designed to simulate simple Spiking Neural Networks (SNNs). Developed under the supervision of Prof. Sandip Lashkare at IIT Gandhinagar, it provides a platform for experimenting with biological neuron models, learning rules, and hardware constraints.

**Documentation:** [https://nervos.readthedocs.io/](https://nervos.readthedocs.io/)

---

### Installation

The simplest way to install `nervos` is using pip:

```bash
pip install nervos
```

### Features

*   **Biologically-Inspired Models:** Implements the Leaky Integrate-and-Fire (LIF) neuron model.
*   **STDP Learning:** Built-in support for Spike-Timing-Dependent Plasticity (STDP) for unsupervised learning.
*   **Hardware Simulation:**
    *   **Synaptic Variability:** Simulate read noise in synaptic weights (`enable_synaptic_noise`).
    *   **Endurance Modeling:** Model cycle-dependent weight updates to simulate device aging or non-idealities (`enable_cycle_dependent_weights`).
*   **Customizable:** Fully configurable parameters for neurons, synapses, and training loops.
*   **Model Persistence:** Save and load trained models and training states.
*   **Data Loaders:** Built-in loaders for MNIST, Iris, and Circles datasets with SNN preprocessing.

### Examples

Check out the [lib_examples](./lib_examples/) directory for usage scripts and notebooks:

1.  **MNIST Classification:** Standard SNN training on MNIST.
2.  **Hardware Variability:** Simulating synaptic noise during inference/training.
3.  **Device Endurance:** Simulating weight update degradation over time.
4.  **Simple Datasets:** Examples for Iris and Circles datasets.
5.  **Single Neuron:** Demonstrations of current injection and spiking behavior.

### Poster & Paper

*   **Poster:** [`./Project_Files/Poster.pdf`](./Project_Files/Poster.pdf)
*   **Paper:** [arXiv](https://arxiv.org/abs/2506.19377) | [Springer](https://link.springer.com/chapter/10.1007/978-3-032-04558-4_41)

### Citation

If you use this library or adapt code from it, please cite the following paper:

```bibtex
@InProceedings{maskeen_unified_2026,
author="Maskeen, Jaskirat Singh
and Lashkare, Sandip",
title="A Unified Platform to Evaluate STDP Learning Rule and Synapse Model Using Pattern Recognition in a Spiking Neural Network",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2025",
year="2026",
publisher="Springer Nature Switzerland",
doi={10.1007/978-3-032-04558-4_41},
address="Cham",
pages="509--520",
isbn="978-3-032-04558-4"
}
```

---
### Feel free to fork and contribute!