"""
This module contains the base `Dataloader` class, which provides the foundation 
for implementing custom dataloaders for various datasets. It primarily serves as 
a blueprint for creating data loading mechanisms.
"""

from ..utils import Parameters


class Dataloader:
    """
    A base class for creating a dataloader.

    This class serves as a blueprint for implementing dataloaders to handle
    data loading tasks. It is initialized with a `Parameters` object containing
    configuration details.

    Attributes:
        parameters (Parameters): An object encapsulating the parameters required
                                 for the dataloader.
    """

    def __init__(self, parameters: Parameters) -> None:
        """
        Initializes the Dataloader with the given parameters.

        Args:
            parameters (Parameters): An object of the `Parameters` class containing
                                     the configuration settings for the dataloader.
        """
        self.parameters = parameters

    def dataloader(self) -> None:
        """
        Placeholder method for loading data.

        This method should be overridden by subclasses to define the specific
        data loading logic.

        Returns:
            None
        """
        return
