"""
Provides the implementation of the Module class for managing and training spiking neural networks (SNNs).
Includes methods for STDP learning, feed-forward processing, training, testing, and model persistence.
"""

from . import common, np, plt
from ..dataloader import Dataloader
from .parameters import Parameters
from .layer import Layer
from .neuron import LIFNeuron
from typing import Union


class Module:
    """
    Represents a module for a spiking neural network.

    This class serves as the main structure to initialize, train, and evaluate a spiking neural network using parameters and layer configurations.
    User must set the necessary data and call methods like `initialise_layers` and `train` to fully utilize the module.

    Attributes:
        parameters (Parameters): Configuration parameters for the spiking neural network, including learning rates and thresholds.
        t0 (float): Timestamp at the initialization of the module, used for tracking execution time.
        str_t0 (str): String representation of the initialization timestamp in the format 'YYYY_MM_DD-HH_MM_SS'.
        dataloader (Dataloader): Placeholder for the data loader object to manage training and testing datasets.
        X_train (np.ndarray): Training input spike trains (initialized as None).
        Y_train (np.ndarray): Training labels corresponding to `X_train` (initialized as None).
        X_test (np.ndarray): Testing input spike trains (initialized as None).
        Y_test (np.ndarray): Testing labels corresponding to `X_test` (initialized as None).
        identifier (str): Unique identifier for the module, defaulting to the timestamp `str_t0` if not provided.
        logger (Logger): Logger object for tracking and debugging operations within the module.
        wta (bool, optional): Used to use Winner-Take-All or update all synapses while doing STDP.
    """

    def __init__(self, parameters: Parameters, identifier: str = None) -> None:
        """
        Initialize the Module object.

        Args:
            parameters (Parameters): Configuration parameters for the Module.
            identifier (str, optional): Unique identifier for the module. Defaults to the current timestamp.
        """
        self.parameters = parameters
        self.t0 = common.timenow()
        self.str_t0 = common.strftime("%Y_%m_%d-%H_%M_%S")
        self.dataloader: Dataloader = None
        self.X_train: np.ndarray = None
        self.Y_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.Y_test: np.ndarray = None
        self.identifier = identifier if identifier else self.str_t0
        self.logger = common.logger
        self.get_spikeplots: bool = False
        self.spikeplots: np.ndarray = []
        self.layerpotentials: np.ndarray = []
        self.get_weight_evolution = False
        self.weight_evolution = []
        self.wta = True
        self.allowed_levels = None

    def STDP(self, delta_t: int) -> float:
        """
        Calculate the spike-timing-dependent plasticity (STDP) adjustment for synapse weights.

        Args:
            delta_t (float): The time difference between post-synaptic and pre-synaptic spikes.

        Returns:
            float: The weight adjustment based on the STDP rule.
        """

        if delta_t >= 0:
            return self.parameters.A_up * (
                np.exp(-float(delta_t) / self.parameters.tau_up)
            )
        else:  # delta_t <0
            return self.parameters.A_down * (
                np.exp(float(delta_t) / self.parameters.tau_down)
            )

    def initialise_layers(self, layer_sizes: list) -> None:
        """
        Initialize the layers of the network based on the specified layer sizes.

        Args:
            layer_sizes (list): A list of integers specifying the number of neurons in each layer.
        """

        self.layer_sizes = layer_sizes
        self.layers = [
            Layer(self.parameters, layer_sizes[j], layer_sizes[j + 1])
            for j in range(len(layer_sizes) - 1)
        ]

    def get_all_synapses(self) -> np.ndarray:
        """
        Retrieve all synapse weights from the network.

        Returns:
            np.ndarray: A list of all synapse weights for each layer in the network.
        """
        return np.array([layer.synapses for layer in self.layers])

    def update_current_potentials_and_adaptive_thresholds(
        self,
        current_potentials: np.ndarray,
        neuron: LIFNeuron,
        neuron_index: int,
        spike_train_at_timestep: np.ndarray,
        synapses_for_neuron_index: np.ndarray,
        time_step: float,
        in_training: bool = False,
    ) -> None:
        """
        Update the membrane potential and adaptive thresholds of a neuron during feed-forward processing.

        Args:
            current_potentials (np.ndarray): Current potentials for all neurons in the layer.
            neuron (LIFNeuron): The neuron to update.
            neuron_index (int): Index of the neuron in the layer.
            spike_train_at_timestep (np.ndarray): Spike train data at the current time step.
            synapses_for_neuron_index (np.ndarray): Synapse weights for the neuron.
            time_step (float): Current time step in the simulation.
            in_training (bool): Is the model in train mode (do_stdp is True).
        """

        if neuron.rest_until < time_step:
            neuron.potential += np.dot(
                synapses_for_neuron_index, spike_train_at_timestep
            )

            if neuron.potential > self.parameters.resting_potential:
                neuron.potential -= self.parameters.spike_drop_rate
                if neuron.adaptive_threshold > self.parameters.spike_threshold:
                    neuron.adaptive_threshold -= self.parameters.threshold_drop_rate

            current_potentials[neuron_index] = neuron.potential
        if in_training:
            neuron.potential_memory[time_step] = neuron.potential
    
    def update_synapse(self, synapse_weight: float, weight_factor: float) -> float:
        """
        Adjust the synapse weight using the specified weight factor.

        Args:
            synapse_weight (float): The current weight of the synapse.
            weight_factor (float): The adjustment factor based on STDP.

        Returns:
            float: The updated synapse weight.
        """

        diff = (
            (synapse_weight - self.parameters.min_weight)
            if weight_factor < 0
            else (self.parameters.max_weight - synapse_weight)
        )
        updated_wt = synapse_weight + self.parameters.eta * weight_factor * (
            np.sign(diff) * abs(diff) ** 0.9
        )
        
        if type(self.allowed_levels) != type(None):
            
            return self.allowed_levels[np.argmin(abs(self.allowed_levels-updated_wt))]
        
        return updated_wt

    def reweigh_synapses_for_between_input_and_output_neuron(
        self,
        current_neuron_idx: int,
        spike_train_for_current_layer: np.ndarray,
        synapses: np.ndarray,
        time_step: float,
        next_neuron_idx: int,
        synapse_memory: np.ndarray,
    ) -> None:
        """
        Apply STDP learning to reweigh synapses connecting input neurons and the output neuron.

        Args:
            current_neuron_idx (int): Index of the input neuron.
            spike_train_for_current_layer (np.ndarray): Spike train for the current layer.
            synapses (np.ndarray): Synapse weights for the layer.
            time_step (float): Current time step in the simulation.
            next_neuron_idx (int): Index of the target output neuron.
            synapse_memory (np.ndarray): Memory to track updated synapses.
        """

        for dt in range(
            0, self.parameters.past_window - 1, -1
        ):  # in past timeframe # -dt is post-pre value
            if 0 <= time_step + dt < self.parameters.training_duration + 1:
                if (
                    spike_train_for_current_layer[current_neuron_idx][time_step + dt]
                    == 1
                ):  # if pre before post
                    synapses[next_neuron_idx][current_neuron_idx] = self.update_synapse(
                        synapses[next_neuron_idx][current_neuron_idx],
                        self.STDP(-dt),
                    )
                    synapse_memory[next_neuron_idx][current_neuron_idx] = 1
        if (
            synapse_memory[next_neuron_idx][current_neuron_idx] != 1
        ):  # if pre not before post in the past timeframe
            synapses[next_neuron_idx][current_neuron_idx] = self.update_synapse(
                synapses[next_neuron_idx][current_neuron_idx],
                self.STDP(
                    np.random.choice(
                        list(range(-1, self.parameters.past_window // 2, -1))
                    )
                ),
            )

    def feed_forward(
        self,
        spike_train: np.ndarray,
        neuron_label_map: np.ndarray,
        training_duration: int,
        do_stdp: bool = False,
        label: int = None,
        spikeplots_for_one_spike_train: list = None,
        layer_potentials_for_one_spike_train: list = None,
        weight_evolution_for_one_spike_train: list = None,
        plotting=False,
    ) -> tuple[np.ndarray, int]:
        """
        Perform feed-forward processing for the Module, with optional STDP learning.

        Args:
            spike_train (np.ndarray): Input spike train.
            neuron_label_map (np.ndarray): Map of neuron indices to labels.
            training_duration (int): Time duration for the feed-forward process.
            do_stdp (bool, optional): If True, apply STDP learning. Defaults to False.
            label (int, optional): Label of the current input. Defaults to None.
            spikeplots_for_one_spike_train (list, optional): Used to store spikeplots for the current input. Defaults to None.
            layer_potentials_for_one_spike_train (list, optional): Used to store potentials of all layers' neurons. Defaults to None.
            plotting (bool, optional): Used to differentiate between testing and training while storing spikeplots.
        Returns:
            tuple: Spike counts for the output layer and the index of the neuron with the highest potential.
        """

        position = 0
        while position < len(self.layer_sizes) - 1:
            if self.get_spikeplots and plotting:
                spikeplots_for_one_spike_train.append(spike_train)
            count_spikes_for_next_layer = np.zeros(self.layer_sizes[position + 1])
            current_potentials_for_next_layer = np.zeros(self.layer_sizes[position + 1])
            synapse_memory = np.zeros(
                (self.layer_sizes[position + 1], self.layer_sizes[position])
            )
            synapses = self.layers[position].synapses
            spike_train_for_next_layer = np.zeros(
                (self.layer_sizes[position + 1], self.parameters.training_duration + 1)
            )
            for time_step in training_duration:
                for neuron_index, neuron in enumerate(self.layers[position].layer):
                    self.update_current_potentials_and_adaptive_thresholds(
                        current_potentials_for_next_layer,
                        neuron,
                        neuron_index,
                        spike_train[:, time_step],
                        synapses[neuron_index],
                        time_step,
                        plotting,
                    )

                highest_potential_neuron_idx = np.argmax(
                    current_potentials_for_next_layer
                )
                highest_potential_neuron = self.layers[position].layer[
                    highest_potential_neuron_idx
                ]

                if (
                    current_potentials_for_next_layer[highest_potential_neuron_idx]
                    < highest_potential_neuron.adaptive_threshold
                ):
                    continue

                count_spikes_for_next_layer[highest_potential_neuron_idx] += 1

                for neuron_index, neuron in enumerate(self.layers[position].layer):
                    if (
                        current_potentials_for_next_layer[neuron_index]
                        > neuron.adaptive_threshold
                    ):
                        spike_train_for_next_layer[neuron_index, time_step] = 1
                        neuron.state_just_after_firing(time_step)
                        if plotting:
                            neuron.potential_memory[time_step] = (
                                neuron.parameters.reset_potential
                            )
                        neuron.adaptive_threshold += 1
                    else:
                        neuron.inhibit(time_step)
                        if plotting:
                            neuron.potential_memory[time_step] = (
                                neuron.parameters.inhibitory_potential
                            )

                if do_stdp:
                    for current_neuron_idx in range(self.layer_sizes[position]):
                        if self.wta:
                            self.reweigh_synapses_for_between_input_and_output_neuron(
                                current_neuron_idx,
                                spike_train,
                                synapses,
                                time_step,
                                highest_potential_neuron_idx,
                                synapse_memory,
                            )
                        else:
                            for next_neuron_idx in range(
                                self.layer_sizes[position + 1]
                            ):
                                self.reweigh_synapses_for_between_input_and_output_neuron(
                                    current_neuron_idx,
                                    spike_train,
                                    synapses,
                                    time_step,
                                    next_neuron_idx,
                                    synapse_memory,
                                )
                    if self.get_weight_evolution and plotting:
                        weight_evolution_for_one_spike_train.append(self.layers[position].synapses.copy())
                        
            self.layers[position].update_neuron_potential_memories()
            if plotting:
                layer_potentials_for_one_spike_train.append(
                    self.layers[position].neuron_potential_memories
                )
            self.layers[position].initial()

            spike_train = spike_train_for_next_layer
            position += 1
        if self.get_spikeplots and plotting:
            spikeplots_for_one_spike_train.append(spike_train)
        if do_stdp and label is not None:
            neuron_label_map[highest_potential_neuron_idx] = int(label)

        return count_spikes_for_next_layer, highest_potential_neuron_idx

    def train(self) -> float:
        """
        Train the Module using the provided dataset.

        Returns:
            float: The accuracy on the test set after training.
        """

        self.logger.info("Training Started")
        self.t0 = common.timenow()
        neuron_label_map = np.repeat(-1, self.layer_sizes[-1])
        training_duration = np.arange(1, self.parameters.training_duration + 1, 1)
        test_accuracy_over_epochs = []
        spikeplots = []
        layerpotentials = []
        weight_evolution = []
        for epoch in range(self.parameters.epochs):
            spikeplots_for_one_epoch = []
            layer_potentials_for_one_epoch = []
            weight_evolution_for_one_epoch = []
            print(f"Epoch {epoch + 1}/{self.parameters.epochs}")
            cnt = 0
            for spike_train, label in zip(self.X_train, self.Y_train):
                cnt += 1
                spikeplots_for_one_spike_train = []
                layer_potentials_for_one_spike_train = []
                weight_evolution_for_one_spike_train = []
                self.feed_forward(
                    spike_train,
                    neuron_label_map,
                    training_duration,
                    do_stdp=True,
                    label=label,
                    spikeplots_for_one_spike_train=spikeplots_for_one_spike_train,
                    layer_potentials_for_one_spike_train=layer_potentials_for_one_spike_train,
                    weight_evolution_for_one_spike_train=weight_evolution_for_one_spike_train,
                    plotting=True,
                )
                spikeplots_for_one_epoch.append(spikeplots_for_one_spike_train)
                layer_potentials_for_one_epoch.append(
                    layer_potentials_for_one_spike_train
                )
                weight_evolution_for_one_epoch.append(weight_evolution_for_one_spike_train)
                bar_length = 40
                progress = int(
                    bar_length * cnt / self.parameters.training_images_amount
                )
                bar = "=" * progress + "." * (bar_length - progress)
                print(
                    f"\r{str(cnt).zfill(len(str(self.parameters.training_images_amount)))}/{self.parameters.training_images_amount} [{bar}]",
                    end="",
                )

            self.learned_synapses = self.get_all_synapses()
            self.learned_neuron_label_map = neuron_label_map
            print("\nTesting...")
            test_accuracy = self.test()
            test_accuracy_over_epochs.append(test_accuracy)
            print(
                f'\rTest set accuracy: {round(test_accuracy, 3) if test_accuracy_over_epochs else "None"}\nTime elapsed since training start: {round(common.timenow() - self.t0, 3)}s'
            )

            self.save_epoch(
                epoch + 1, self.learned_synapses, neuron_label_map, test_accuracy
            )
            spikeplots.append(spikeplots_for_one_epoch)
            layerpotentials.append(layer_potentials_for_one_epoch)
            weight_evolution.append(weight_evolution_for_one_epoch)

        self.spikeplots = spikeplots
        self.layerpotentials = layerpotentials
        self.weight_evolution = weight_evolution

        print("Training complete!")
        return test_accuracy

    def test(self) -> float:
        """
        Test the Module on the provided test dataset.

        Returns:
            float: The accuracy of the Module on the test dataset.
        """

        t = 0
        c = 0
        for spike_train, label in zip(self.X_test, self.Y_test):
            prediction = self.get_prediction(spike_train)
            if prediction == label:
                c += 1
            t += 1
        return c / t

    def get_prediction(
        self,
        spike_train: np.ndarray,
        all_synapses: np.ndarray = None,
        neuron_label_map: np.ndarray = None,
    ) -> int:
        """
        Predict the label of a given input spike train.

        Args:
            spike_train (np.ndarray): Input spike train.
            all_synapses (np.ndarray, optional): Synapse weights. Defaults to learned synapses.
            neuron_label_map (np.ndarray, optional): Neuron-to-label mapping. Defaults to learned mapping.

        Returns:
            int: Predicted label.
        """

        if not (
            isinstance(all_synapses, np.ndarray)
            and isinstance(neuron_label_map, np.ndarray)
        ):
            synapses = self.learned_synapses.copy()
            neuron_label_map = self.learned_neuron_label_map.copy()
        else:
            synapses = all_synapses.copy()
            neuron_label_map = neuron_label_map.copy()

        for i in range(len(self.layers)):
            self.layers[i].set_synapses(synapses[i])

        training_duration = np.arange(1, self.parameters.training_duration + 1, 1)
        count_spikes_for_output_layer, _ = self.feed_forward(
            spike_train,
            neuron_label_map,
            training_duration,
            do_stdp=False,
            plotting=False,
        )
        prediction = neuron_label_map[np.argmax(count_spikes_for_output_layer)]
        return prediction

    def predict(self, X: np.ndarray, model_location: str) -> int:
        """
        Predict labels for a batch of inputs. User has to implement this on their own.

        Args:
            X (np.ndarray): Batch of input spike trains.
            model_location (str): Path to the model for inference.

        Returns:
            int: Predicted labels for the input batch.
        """

        return -1

    def save_epoch(
        self,
        epoch: int,
        synapses: np.ndarray,
        neuron_label_map: np.ndarray,
        accuracy: float,
    ) -> None:
        """
        Save the model after a training epoch.

        Args:
            epoch (int): Current epoch number.
            synapses (np.ndarray): Synapse weights after the epoch.
            neuron_label_map (np.ndarray): Neuron-to-label mapping.
            accuracy (float): Accuracy on the test set after the epoch.
        """

        common.mkdir(f"{common.cwd}\\storage")
        common.mkdir( 
            f"{common.cwd}\\storage\\{self.identifier}{'_'+self.str_t0 if self.str_t0!=self.identifier else ''}"
        )

        common.mkdir(
            f"{common.cwd}\\storage\\{self.identifier}{'_'+self.str_t0 if self.str_t0!=self.identifier else ''}\\Epoch_{epoch}-{round(accuracy,3)}"
        )

        model_path = f"{common.cwd}\\storage\\{self.identifier}{'_'+self.str_t0 if self.str_t0!=self.identifier else ''}\\Epoch_{epoch}-{round(accuracy,3)}\\model.red"

        common.save_model(neuron_label_map, synapses, self.parameters, model_path)

    def load_model(self, model_location: str) -> None:
        """
        Load a saved SNN model from the specified location.

        Args:
            model_location (str): Path to the saved model.
        """

        synapses, neuron_label_map, parameters = common.load_model(model_location)
        self.learned_synapses = synapses
        self.learned_neuron_label_map = neuron_label_map
        self.parameters = parameters
