from abc import ABC, abstractmethod
from pathlib import Path

import jax.numpy as jnp
from flax import linen as nn
from jax import Array
from optax import GradientTransformation, OptState

from Common import globals
from Visualization import memory_visualization


class MemoryInterface(ABC, nn.Module):
    @abstractmethod
    def __init__(
        self, memory_shape: tuple[int, ...], optimizer: GradientTransformation
    ):
        self.optimizer: GradientTransformation
        self.weights: Array
        self.optimizer_state: OptState

    @abstractmethod
    def apply_gradients(self, gradients) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, memory_weights: Array, read_weights: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        memory_weights: Array,
        read_weights: Array,
        erase_vector: Array,
        add_vector: Array,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def address(
        self,
        memory_weights: Array,
        key_vector: Array,
        key_strength: Array,
        interp_gate_scalar: Array,
        shift_weights: Array,
        sharpen_scalar: Array,
        previous_weights: Array,
    ) -> Array:
        raise NotImplementedError


# TODO options for:
# save location for plots?
# selecting at what points plots should be made (read/write/address, read key)
# annotations (title, etc) to the plots
class MemoryVisualizerWrapper(MemoryInterface):
    """Wrapper around memory to make plots during memory functions.

    Usage:
    memory = Memory()
    memory = MemoryVisualizerWrapper(memory)
    """

    def __init__(
        self,
        wrapped_memory: MemoryInterface,
        save_dir: str | None = None,
        save_name: str | None = None,
    ):
        self.wrapped_memory = wrapped_memory
        self.weights = self.wrapped_memory.weights
        self.save_dir: str = save_dir if save_dir else ""
        self.save_name: str = (
            save_name if save_name else globals.VISUALIZATION.NAMES.DEFAULT
        )

    def apply_gradients(self, gradients) -> None:
        return self.wrapped_memory.apply_gradients(gradients)

    def read(self, memory_weights, read_weights):
        memory_visualization.plot_memory_state(
            memory_weights.squeeze(0).transpose(),
            read_weights.squeeze(0),
            save_location=self.get_save_path(globals.VISUALIZATION.NAMES.READ),
        )

        read_output = self.wrapped_memory.read(memory_weights, read_weights)
        return read_output

    def write(self, memory_weights, write_weights, erase_vector, add_vector):
        memory_visualization.plot_memory_state(
            memory_weights.squeeze(0).transpose(),
            write_weights.squeeze(0),
            save_location=self.get_save_path(globals.VISUALIZATION.NAMES.WRITE),
        )

        write_output = self.wrapped_memory.write(
            memory_weights, write_weights, erase_vector, add_vector
        )

        return write_output

    def address(
        self,
        memory_weights,
        key_vector,
        key_strength,
        interp_gate_scalar,
        shift_weights,
        sharpen_scalar,
        previous_weights,
    ):
        address_output = self.wrapped_memory.address(
            memory_weights,
            key_vector,
            key_strength,
            interp_gate_scalar,
            shift_weights,
            sharpen_scalar,
            previous_weights,
        )

        memory_visualization.plot_memory_state(
            memory_weights.squeeze(0).transpose(),
            address_output.squeeze(0),
            save_location=self.get_save_path(globals.VISUALIZATION.NAMES.ADDRESS),
        )

        return address_output

    def get_save_path(self, vis_type: str, leading_zeros=2):
        counter = 0
        base_path = str(Path(self.save_dir) / f"{self.save_name}_{vis_type}")

        max_counter = 10**leading_zeros
        while counter < max_counter:
            test_path = f"{base_path}_{str(counter).zfill(leading_zeros)}"
            test_path_real = memory_visualization.get_save_path(
                test_path, globals.VISUALIZATION.IMG_EXTENSION
            )
            if not Path.is_file(Path(test_path_real)):
                break
            counter += 1

        assert (
            counter != max_counter
        ), "Memory visualization wrapper ran out of filenames."

        return test_path


class MemoryStub(MemoryInterface):
    def __init__(self, N, M):
        self.N = N
        self.M = M

    def size(self):
        return self.N, self.M

    def read(self, read_weights):
        """Return the input (length N)"""
        return read_weights

    def address(
        self,
        key_vector,
        key_strength,
        interp_gate_scalar,
        shift_weights,
        sharpen_scalar,
        previous_weights,
    ):
        """Return key_vector (length M) padded to be of length N"""
        return jnp.pad(
            key_vector, self.N - self.M
        )  # note that jnp.pad allows negative padding. pad value defaults to 0s
