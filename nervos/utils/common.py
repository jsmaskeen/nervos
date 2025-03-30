"""
This common module provides helper functions and tools for various tasks such as 
logging, file and directory operations, time handling, and model saving/loading 
using pickle. It serves as a general-purpose utility module for the Nervos project.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import pickle


def timenow() -> float:
    """
    Get the current time as a Unix timestamp.

    Returns:
        float: The current time in seconds since the epoch.
    """
    return time.time()


def strftime(t: str) -> str:
    """
    Format the given time string.

    Args:
        t (str): A time format string (e.g., "%Y-%m-%d %H:%M:%S").

    Returns:
        str: The formatted time string.
    """
    return time.strftime(t)


logger = logging.getLogger("Nervos")


def get_cwd() -> str:
    """
    Get the current working directory.

    Returns:
        str: The path of the current working directory.
    """
    return os.getcwd()


cwd = get_cwd()


def mkdir(f: str) -> None:
    """
    Create a directory if it doesn't already exist.

    Args:
        f (str): The path of the directory to create.

    Returns:
        None
    """
    if os.path.exists(f):
        return
    os.mkdir(f)


def save_model(labels: np.ndarray, synapses: np.ndarray, parameters:dict, path: str) -> None:
    """
    Save model data (labels and synapses) to a file.

    Args:
        labels (np.ndarray): Array of labels associated with the model.
        synapses (np.ndarray): Array representing the synapses of the model.
        parameters (dict): Dictionary of parameters used.
        path (str): The file path to save the model.

    Returns:
        None

    Example:
        save_model(labels, synapses, "model.red")
    """
    data = {"labels": labels, "synapses": synapses,"parameters":parameters}

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_model(path:str)->tuple[np.ndarray,np.ndarray,dict]:
    """
    Load model data (labels and synapses) from a file.

    Args:
        path (str): The file path of the saved model.

    Returns:
        tuple: A tuple containing:
            - synapses (np.ndarray): Array representing the synapses of the model.
            - labels (np.ndarray): Array of labels associated with the model.
            - parameters (dict): Dictionary of parameters used.

    Example:
        synapses, labels, parameters = load_model("model.red")
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["synapses"], data["labels"],data["parameters"]

def plot_spike_train(spike_train:np.ndarray,ylim:tuple=None) -> None:
    """
    Plot spike trains
    
    Args:
        spike_train (np.ndarray): Spike trains to plot.
        ylim (tuple, optional): y limit of the graph.
    """
    plt.figure(figsize=(10, 6))
    for i, spikes in enumerate(spike_train):
        spike_times = np.where(spikes == 1)[0]
        plt.scatter(spike_times, [i] * len(spike_times), marker="|", color="k")

    plt.xlabel("Time Steps")
    plt.ylabel("Neurons")
    plt.title("Spike Train Raster Plot")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    if ylim:
        plt.ylim(ylim)
        plt.yticks(range(0,int(ylim[1]+1)))
    plt.show()