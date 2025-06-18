"""
Nervos: A Spiking Neural Network Simulation Framework
=====================================================

This package provides tools for simulating spiking neural networks (SNNs), including components for neuron models, layers, synapses, and training processes. The framework is designed to model spiking dynamics and facilitate experimentation with biological neuron models like the Leaky Integrate-and-Fire (LIF) neuron.

Modules:
--------
    - common: Contains utility functions and global configurations for the simulation, such as time tracking, logging, and model saving/loading.
    - layer: Defines the Layer class, which represents a collection of neurons and their synaptic connections in a layer of the network.
    - module: Contains the Module class, which models the overall network, including layer management, synapse updates, and spike-timing-dependent plasticity (STDP).
    - neuron: Defines the LIFNeuron class and other neuron models, including their behavior, states, and interactions with synaptic inputs.
    - parameters: Contains the Parameters class that holds various configuration settings for the network, such as simulation duration, learning rates, and neuron properties.

Examples:
---------
    - ExampleCurrentThroughLIFNeuron: Demonstrates simulating the behavior of an LIF neuron with an input current pulse and visualizing the resulting membrane potential and spikes.

Usecase:
--------
    - The Nervos package allows for creating, training, and testing spiking neural networks, simulating spike-timing-dependent plasticity, and visualizing the results of neuronal activities.
"""

from .utils import *
from .examples import *
from .dataloader import *