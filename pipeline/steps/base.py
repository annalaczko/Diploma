"""Base abstract class for all pipeline steps."""

from abc import ABC, abstractmethod
from typing import Any


class Step(ABC):
    """Abstract base class for every stage in the pipeline
    (preprocessing, feature learning, clustering, evaluation).

    Each concrete step must implement `run`, which takes the
    input data and returns the processed output. Subclasses can
    add their own config via __init__.
    """

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Execute the step on the given data and return the result."""
        raise NotImplementedError

    # @abstractmethod
    # def save_data(self, path: str) -> None:
    #     """Save the step's state to the specified path."""
    #     raise NotImplementedError

    def __call__(self, data: Any) -> Any:
        # allows step_instance(data) as a shortcut for step_instance.run(data)
        return self.run(data)

    def __repr__(self) -> str:
        return f"<{self.name}>"
