"""
This package implements a spiking neural network (SNN) framework, which includes modules for neurons, layers, parameters, and training.

The package provides a structure for creating and training a spiking neural network using biologically inspired spiking neurons, such as the Leaky Integrate-and-Fire (LIF) neuron model, and implements spike-timing dependent plasticity (STDP) for learning. It is composed of multiple modules and helper functions to support various operations like layer initialization, network training, data handling, and parameter management.

Submodules:
    - common.py: Provides utility functions for logging, time tracking, file handling, and model saving/loading.
    - layer.py: Contains the `Layer` class, which represents a layer in the network. It manages the list of neurons and synapses between them.
    - module.py: Implements the main `Module` class, which manages the overall spiking neural network, including training, testing, and synapse updates.
    - neuron.py: Defines the `LIFNeuron` class for the Leaky Integrate-and-Fire neurons, including mechanisms for spiking, threshold adaptation, and potential updates.
    - parameters.py: Defines the `Parameters` class for configuring and storing network parameters, including learning rates, synapse properties, and training settings.

Key Features:
    - Spiking Neurons: The `LIFNeuron` class models a biological spiking neuron, including mechanisms for potential integration, spiking, and adaptation.
    - STDP (Spike-Timing Dependent Plasticity): The `STDP` method in the `Module` class implements the biological learning rule where synapse weights are adjusted based on spike timing between pre- and post-synaptic neurons.
    - Layer and Synapse Management: The `Layer` class allows for layer-wise organization of neurons and synapses, enabling flexible model architecture.
    - Training and Testing: The `Module` class facilitates network training and testing, managing data, synapse updates, and performance evaluation over multiple epochs.
    - Parameter Management: The `Parameters` class handles network configuration, including saving and loading parameters from various sources (files, URLs, or dictionaries).

This package is designed for research and experimentation with spiking neural networks and is particularly useful for understanding biological learning principles in artificial systems.
"""

from .common import *
from .parameters import Parameters
from .neuron import LIFNeuron
from .module import Module
from .layer import Layer