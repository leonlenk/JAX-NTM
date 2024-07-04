from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array


class ControllerInterface(ABC, nn.Module):
    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def register_parameters(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def is_read_controller(self) -> bool:
        raise NotImplementedError
