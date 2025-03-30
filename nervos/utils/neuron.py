"""
This module defines the `LIFNeuron` class, which models a Leaky Integrate-and-Fire (LIF) neuron.
The neuron is characterized by its potential dynamics, refractory period, and adaptive threshold.
It provides methods for updating the neuron's state after firing or being inhibited, 
as well as resetting its initial state.
"""

from .parameters import Parameters


class LIFNeuron:
    """
    A Leaky Integrate-and-Fire (LIF) neuron model.

    This class models the dynamics of a LIF neuron, including its potential updates,
    refractory behavior, and adaptive threshold.

    Attributes:
        parameters (Parameters): Configuration parameters for the neuron.
        adaptive_threshold (float): The dynamic threshold for the neuron to fire.
        reftactory_time (float): The refractory time during which the neuron cannot fire.
        potential (float): The current membrane potential of the neuron.
        rest_until (int): The time step until which the neuron remains at rest.
        potential_memory dict[int,float]: Stores the potential at any time step of training.
    """

    def __init__(self, parameters: Parameters) -> None:
        """
        Initialize a new LIFNeuron instance with the given parameters.

        Args:
            parameters (Parameters): Configuration parameters for the neuron.
        """
        self.parameters = parameters
        self.adaptive_threshold:float = None
        self.reftactory_time:float = None
        self.potential:float = None
        self.rest_until:float = None
        self.potential_memory:dict[int,float] = {}
        self.initial()

    def state_just_after_firing(self, time_step: float):
        """
        Update the neuron's state immediately after firing.

        Sets the potential to the reset potential and schedules the neuron
        to remain at rest until the end of the refractory period.

        Args:
            time_step (flaot): The current time step when the neuron fires.
        """
        self.potential = self.parameters.reset_potential
        self.rest_until = time_step + self.reftactory_time

    def inhibit(self, time_step: float):
        """
        Inhibit the neuron by setting its potential to the inhibitory potential.

        Args:
            time_step (float): The current time step when the neuron is inhibited.
        """
        self.potential = self.parameters.inhibitory_potential
        self.rest_until = time_step + self.reftactory_time

    def initial(self):
        """
        Reset the neuron's state to its initial configuration.

        Initializes the adaptive threshold, refractory time, resting potential,
        and resets the `rest_until` attribute.
        """
        self.adaptive_threshold = self.parameters.spike_threshold
        self.rest_until = -1
        self.reftactory_time = self.parameters.refractory_time
        self.potential = self.parameters.resting_potential
        self.potential_memory = {}
