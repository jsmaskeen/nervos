from ..utils.neuron import LIFNeuron
from ..utils.common import *
import matplotlib.gridspec as gridspec

class ExampleCurrentThroughLIFNeuron:
    """
    Example demonstrating the simulation of current passing through a Leaky Integrate-and-Fire (LIF) neuron.

    The class simulates the behavior of a single LIF neuron under an input current pulse and visualizes the resulting membrane potential and input current over time.

    Attributes:
        parameters (Parameters): Configuration parameters for the neuron, such as resting potential, spike threshold, and conductance.
        neuron (LIFNeuron): Instance of the LIFNeuron class, representing the spiking neuron.
        dt (float): Time step (in ms) used for the simulation.
        simulation_duration (int): Total duration (in ms) for the simulation.
        clock (np.ndarray): Time vector that spans the duration of the simulation.

    Methods:
        simulate_pulse(current_magnitude): Simulates the response of the LIF neuron to a current pulse, calculates membrane potential over time, and plots the results.
            current_magnitude (float): The magnitude of the current pulse (in nA) applied to the neuron.
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.neuron = LIFNeuron(parameters)
        self.dt = 0.1
        self.simulation_duration = 500
        self.clock = np.arange(0, self.simulation_duration, self.dt)

    def simulate_pulse(self, current_magnitude):
        V = np.zeros(self.clock.size)
        V[0] = self.parameters.resting_potential
        I = current_magnitude * np.ones(self.clock.size)
        I[: int(self.clock.size // 2 - 1e3)] = 0
        I[int(self.clock.size // 2 + 1e3) :] = 0

        spike_time_index = []
        refractory_counter = 0

        for time_step in range(self.clock.size - 1):
            if refractory_counter > 0:
                V[time_step] = self.parameters.resting_potential
                refractory_counter -= 1
            elif V[time_step] >= self.parameters.spike_threshold:
                spike_time_index.append(time_step)
                V[time_step] = self.parameters.resting_potential
                refractory_counter = self.parameters.refractory_time / self.dt

            dV = (
                (
                    -(V[time_step] - self.parameters.resting_potential)
                    + I[time_step] / self.parameters.conductance
                )
                * self.dt
                / self.parameters.tau_m
            )
            V[time_step + 1] = V[time_step] + dV

        if len(spike_time_index) > 0:
            V[np.array(spike_time_index) - 1] += 10
        plt.figure(figsize=(7, 5))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 0, 1])
        ax1 = plt.subplot(gs[0:2])
        ax1.plot(self.clock, V, "r", label="Membrane\npotential")
        ax1.axhline(
            self.parameters.spike_threshold,
            0,
            1,
            color="k",
            ls="--",
            label="Threshold $V_{th}$",
        )
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("V (mV)")
        ax1.legend(loc=[1.05, 0.75])
        ax1.set_ylim([-80, -40])
        ax2 = plt.subplot(gs[2])
        ax2.plot(self.clock, I, label="Input Current")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Current")
        ax2.legend(loc=[1.05, 0.75])
        plt.tight_layout()
        plt.show()
