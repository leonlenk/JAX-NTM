from abc import ABC, abstractmethod

from jax import Array


class ControllerInterface(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def create_new_state(self, batch_size: int) -> Array:
        pass

    @abstractmethod
    def register_parameters(self) -> Array:
        pass

    @abstractmethod
    def is_read_controller(self) -> bool:
        pass
