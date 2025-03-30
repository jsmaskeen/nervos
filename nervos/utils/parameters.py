"""
This module defines the `Parameters` class, which serves as a flexible container for various configuration parameters. 
The class provides methods for loading parameters from a URL, dictionary, or file, listing all parameters, 
and saving them to a JSON file.
"""

import json
import requests
from . import common


class Parameters:
    """
    A dynamic configuration manager for storing, updating, and persisting parameters.

    This class allows parameters to be loaded from different sources (URL, dictionary, file),
    listed in a readable format, or saved to a JSON file.

    Methods:
        from_url(url): Load parameters from a JSON response at a given URL.
        from_dict(parameters): Load parameters from a dictionary.
        from_file(path): Load parameters from a JSON file.
        list_parameters(): Print all parameters in a formatted list.
        save(identifier, directory): Save the parameters to a JSON file.
    """

    def __init__(self) -> None:
        """Initialize an empty Parameters instance."""
        pass

    def from_url(self, url: str) -> None:
        """
        Load parameters from a JSON response at the specified URL.

        Args:
            url (str): The URL from which to fetch the JSON parameters.
        """
        parameters = requests.get(url).json()
        for k, v in parameters.items():
            setattr(self, k, v)

    def from_dict(self, parameters: dict) -> None:
        """
        Load parameters from a dictionary.

        Args:
            parameters (dict): A dictionary of parameters to set.
        """
        for k, v in parameters.items():
            setattr(self, k, v)

    def from_file(self, path: str) -> None:
        """
        Load parameters from a JSON file.

        Args:
            path (str): The path to the JSON file containing parameters.
        """
        with open(path, "r") as f:
            parameters = json.load(f)
        for k, v in parameters.items():
            setattr(self, k, v)

    def list_parameters(self) -> None:
        """
        Print all parameters and their values in a formatted table.
        """
        for k, v in self.__dict__.items():
            print(f"{k}:{' '*(30-len(k))}{v}")

    def save(
        self, identifier: str, directory: str = f"{common.cwd}\\parameters"
    ) -> None:
        """
        Save the parameters to a JSON file.

        Args:
            identifier (str): A unique identifier for the saved file.
            directory (str, optional): The directory in which to save the file. Defaults to 'parameters' in the current working directory.
        """
        common.mkdir(directory)
        with open(f"{directory}\\parameters_{identifier}.json", "w") as f:
            f.write(json.dumps(self.__dict__, indent=4))
