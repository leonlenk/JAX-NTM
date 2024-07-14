import os
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from Common import globals
from Common.globals import DATASETS


class DataEncoderInterface(ABC):
    @abstractmethod
    def __init__(self, memory_depth: int):
        """Initializes the encoder.

        :param memory_depth: length of vector at each memory location
        """
        pass

    # TODO can we set this up so we just pass a reference to the DataEncoder class into the DataLoader
    # and instantiate the DataEncoder class inside there? Will avoid the duplicate memory_depth argument needed here.
    @abstractmethod
    def initialize(self, config: dict = {}):
        """Secondary encoder initialization.
        Called when the Dataloader is initialized, through initialize_dataset.
        Executes Dataloader-specific operations.

        :param config: Config options. Populated with keys from Common.globals.DATASETS.ENCODERS.CONFIGS
        """
        pass

    @abstractmethod
    def save(self):
        """Saves the current encoder to cache"""
        pass

    @abstractmethod
    def encode(self, data: Any) -> Array:
        """Encodes data.

        :param data: the data to convert to memory values
        """
        pass

    @abstractmethod
    def decode(self, memory: Array) -> Any:
        """Decodes memory.

        :param data: the memory values to convert to data
        """
        pass


class CurriculumSchedulerInterface(ABC):
    @abstractmethod
    def __init__(self, config: dict = {}):
        """Initializes the curriculum scheduler.

        :param config: a set of configuration options custom to the type of scheduler. Populated with keys from Common.globals.CURRICULUM.CONFIGS
        """
        pass

    @abstractmethod
    def update_curriculum_level(self, curriculum_params: dict):
        """Updates the curriculum level based on metrics.

        :param curriculum_params: inputs influencing next curriculum level. Populated with keys from Common.globals.CURRICULUM.PARAMS
        """
        pass

    @abstractmethod
    def get_curriculum(self, batch_size: int) -> Array:
        """Gets a curriculum (1D array of difficulties) for a single batch.

        :param batch_size: length of curriculum to select
        """
        pass


class DataloaderInterface(ABC):
    def __init__(
        self,
        batch_size: int,
        num_batches: int,
        memory_depth: int,
        seed: int = globals.JAX.RANDOM_SEED,
        config: dict = {},
    ):
        """Initializes the dataloader.
        Runs "self.initialize_dataset" on init.

        :param batch_size: number of samples per batch
        :param num_batches: number of batches in the dataset
        :param memory_depth: length of vector at each memory location
        :param seed: random seed for generating/ordering data
        :param config: options custom to dataset
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.memory_depth = memory_depth
        self.prng = jax.random.key(seed)
        self.config = config

        self.curriculum_scheduler: CurriculumSchedulerInterface = config.get(
            DATASETS.CONFIGS.CURRICULUM_SCHEDULER, CurriculumSchedulerStub()
        )
        self.data_encoder: DataEncoderInterface = config.get(
            DATASETS.CONFIGS.DATA_ENCODER, DataEncoderStub(0)
        )

        self.iterations = 0

        self.common_directory = os.path.abspath(os.path.dirname(__file__) + "/..")
        self.cache_dir = os.path.join(self.common_directory, DATASETS.CACHE_LOCATION)

        # initialize the dataset if necessary
        self.initialize_dataset()

        pass

    def initialize_dataset(self):
        """To be overridden as necessary"""
        self.data_encoder.initialize()
        pass

    def update_batch_params(
        self, batch_size: int | None = None, num_batches: int | None = None
    ):
        """Updates the dataloader batch parameters.

        :param batch_size: number of samples per batch
        :param num_batches: number of batches in the dataset
        """
        if batch_size:
            self.batch_size = batch_size
        if num_batches:
            self.num_batches = num_batches
        pass

    def __iter__(self):
        """Resets the iteration count."""
        self.iterations = 0
        return self

    def __len__(self):
        """Returns the number of batches in the dataset."""
        return self.num_batches

    @abstractmethod
    def update_curriculum_level(self, curriculum_params: dict):
        """Updates the curriculum level based on metrics.

        :param curriculum_params: inputs influencing next curriculum level. Populated with keys from Common.globals.CURRICULUM.PARAMS
        """
        pass

    @abstractmethod
    def update_split(self, new_split: str):
        """Changes the split (train / test / val) used to pull data from.

        :param new_split: The split of the data to switch to
        """
        pass

    @abstractmethod
    def __next__(self) -> tuple[jax.Array, jax.Array]:
        """Gets (or creates) the next batch in the dataset."""
        pass


class DataEncoderStub(DataEncoderInterface):
    def __init__(self, memory_depth: int):
        pass

    def initialize(self, config: dict = {}):
        pass

    def save(self):
        pass

    def encode(self, data: Any) -> Array:
        return jnp.array([])

    def decode(self, memory: Array) -> Any:
        return None


class CurriculumSchedulerStub(CurriculumSchedulerInterface):
    """Stub CurriculumScheduler which always returns difficulties of 1"""

    def __init__(self, config: dict = {}):
        self.config = config

    def update_curriculum_level(self, curriculum_params: dict):
        pass

    def get_curriculum(self, batch_size: int) -> Array:
        return jnp.ones((batch_size,))
