import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from Common.globals import METADATA, VISUALIZATION
from Visualization import memory_visualization


class MemoryInterface(ABC, nn.Module):
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

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
        }


# TODO options for:
# save location for plots?
# selecting at what points plots should be made (read/write/address, read key)
# annotations (title, etc) to the plots
class MemoryVisualizerWrapper(MemoryInterface):
    """Wrapper around memory to make plots during memory functions.

    Usage:
    memory = Memory()
    memory = MemoryVisualizerWrapper(memory, ...)
    """

    def __init__(
        self,
        wrapped_memory: MemoryInterface,
        save_dir: str | None = None,
        save_name: str | None = None,
        delete_existing: bool = False,
    ):
        self.wrapped_memory = wrapped_memory
        self.save_dir: Path = Path(save_dir) if save_dir else Path("")

        if not self.save_dir.is_absolute():
            self.save_dir = Path(VISUALIZATION.OUTPUT_DIR) / self.save_dir
            self.save_dir = self.save_dir.resolve()

        self.save_name: str = save_name if save_name else VISUALIZATION.NAMES.DEFAULT

        if delete_existing:
            if self.save_dir.is_dir():
                shutil.rmtree(str(self.save_dir))

    def read(self, memory_weights, read_weights):
        memory_visualization.plot_memory_state(
            memory_weights.transpose(),
            read_weights,
            save_location=self.get_save_path(VISUALIZATION.NAMES.READ),
            annotation="Read Attention",
        )

        read_output = self.wrapped_memory.read(memory_weights, read_weights)
        return read_output

    def write(self, memory_weights, write_weights, erase_vector, add_vector):
        memory_visualization.plot_memory_state(
            memory_weights.transpose(),
            write_weights,
            save_location=self.get_save_path(VISUALIZATION.NAMES.WRITE),
            annotation="Write Attention",
        )

        write_output = self.wrapped_memory.write(
            memory_weights, write_weights, erase_vector, add_vector
        )

        memory_visualization.plot_memory_state_comparison(
            memory_weights.transpose(),
            write_output.transpose(),
            save_location=self.get_save_path(VISUALIZATION.NAMES.WRITE),
            annotation=["before write", "after write"],
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
            memory_weights.transpose(),
            address_output,
            save_location=self.get_save_path(VISUALIZATION.NAMES.ADDRESS),
        )

        return address_output

    def get_save_path(self, vis_type: str, leading_zeros=2):
        counter = 0
        base_path = str(self.save_dir / f"{self.save_name}_{vis_type}")

        max_counter = 10**leading_zeros
        while counter < max_counter:
            test_path = f"{base_path}_{str(counter).zfill(leading_zeros)}{VISUALIZATION.IMG_EXTENSION}"
            if not Path.is_file(Path(test_path)):
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
