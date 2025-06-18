"""
This module defines the `Layer` class, which represents a single layer of neurons 
in a neural network. It uses the `LIFNeuron` class to model neurons and provides 
methods for initializing and managing synaptic weights. The layer is designed to 
be flexible and configurable through the `Parameters` object.
"""

from . import common, np
from .neuron import LIFNeuron
from .parameters import Parameters


class Layer:
    """
    Represents a single layer of neurons in a neural network.

    Each layer consists of `LIFNeuron` instances and synaptic connections
    that define the interactions between input and output neurons.

    Attributes:
        parameters (Parameters): Configuration parameters for the layer.
        layer (list[LIFNeuron]): List of neurons in the layer.
        synapses (np.ndarray): Synaptic weight matrix (output neurons x input neurons).
        num_input_neurons (int): Number of input neurons to the layer.
        num_output_neurons (int): Number of output neurons in the layer.
        neuron_potential_memories (list): List containing potentials of all neurons in the layer.
    """

    def __init__(
        self, parameters: Parameters, num_input_neurons: int, num_output_neurons: int
    ) -> None:
        """
        Initialize a new Layer instance.

        Args:
            parameters (Parameters): Configuration parameters for the neurons.
            num_input_neurons (int): Number of input neurons to the layer.
            num_output_neurons (int): Number of output neurons in the layer.
        """
        self.parameters = parameters
        self.layer = [LIFNeuron(parameters) for _ in range(num_output_neurons)]
        self.synapses = np.ones((num_output_neurons, num_input_neurons)) #np.random.random((num_output_neurons, num_input_neurons)) #
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.neuron_potential_memories:list = []

    def initial(self) -> None:
        """
        Initialize all neurons in the layer.

        Calls the `initial` method of each `LIFNeuron` in the layer to reset
        their internal states.
        """
        for neuron in self.layer:
            neuron.initial()
        self.neuron_potential_memories = []

    def set_synapses(self, synapses: np.ndarray) -> None:
        """
        Set the synaptic weights for the layer.

        Args:
            synapses (np.ndarray): A weight matrix of shape
                                   (num_output_neurons, num_input_neurons).
        """
        self.synapses = synapses
        
    def update_neuron_potential_memories(self) ->None:
        """
        Update the layer's neurons' potential memory list.
        """
        self.neuron_potential_memories = []
        for neuron in self.layer:
            self.neuron_potential_memories.append(neuron.potential_memory)
            
