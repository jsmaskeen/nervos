"""
This package provides examples for simulating the behavior of spiking neural networks.

The included example demonstrates how to model and simulate the response of a Leaky Integrate-and-Fire (LIF) neuron to a current pulse. The simulation tracks the neuron's membrane potential over time and visualizes the results.

Modules:
    - ExampleCurrentThroughLIFNeuron: Class that simulates the behavior of an LIF neuron under an input current pulse, and generates plots of membrane potential and input current.
"""

from .neuron_spiking import *
