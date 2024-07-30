import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import core
from flax.training.train_state import TrainState
from jax import Array

from Common import globals
from Common.BackboneInterface import BackboneInterface
from Common.globals import CONFIG, DATASETS, METADATA
from Common.MemoryInterface import MemoryInterface


class DataEncoderInterface(ABC):
    memory_depth: int

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

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            METADATA.MEMORY_DEPTH: self.memory_depth,
        }


class CurriculumSchedulerInterface(ABC):
    config: dict = {}

    @abstractmethod
    def __init__(self, initial_curriculum_level: int = 0, config: dict = {}):
        """Initializes the curriculum scheduler.

        :param config: a set of configuration options custom to the type of scheduler. Populated with keys from Common.globals.CURRICULUM.CONFIGS
        """
        self.curriculum_level = initial_curriculum_level
        pass

    @abstractmethod
    def update_curriculum_level(self, curriculum_params: dict):
        """Updates the curriculum level based on metrics.

        :param curriculum_params: inputs influencing next curriculum level. Populated with keys from Common.globals.CURRICULUM.PARAMS
        """
        pass

    @abstractmethod
    def get_curriculum_level(self) -> int:
        return self.curriculum_level

    @abstractmethod
    def get_curriculum(self, batch_size: int) -> Array:
        """Gets a curriculum (1D array of difficulties) for a single batch.

        :param batch_size: length of curriculum to select
        """
        pass

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            METADATA.CONFIG: self.config,
        }


class DataloaderConfig:
    def __init__(
        self,
        curriculum_scheduler: CurriculumSchedulerInterface | None = None,
        data_encoder: DataEncoderInterface | None = None,
        split: str = DATASETS.SPLITS.TRAIN,
        single_level: bool = True,
        accuracy_tolerance: float = DATASETS.DEFAULT_ACCURACY_TOLERANCE,
    ):
        self.curriculum_scheduler: CurriculumSchedulerInterface = (
            curriculum_scheduler
            if curriculum_scheduler is not None
            else CurriculumSchedulerStub()
        )
        self.data_encoder: DataEncoderInterface = (
            data_encoder if data_encoder is not None else DataEncoderStub(0)
        )
        self.single_level = single_level
        self.accuracy_tolerance = accuracy_tolerance
        self.split = split


class DataloaderInterface(ABC):
    batch_size: int
    num_batches: int
    memory_depth: int
    seed: int = globals.JAX.RANDOM_SEED
    options: dict = {}

    def __init__(
        self,
        batch_size: int,
        num_batches: int,
        memory_depth: int,
        config: DataloaderConfig,
        seed: int = globals.JAX.RANDOM_SEED,
        options: dict = {},
    ):
        """Initializes the dataloader.
        Runs "self.initialize_dataset" on init.

        :param batch_size: number of samples per batch
        :param num_batches: number of batches in the dataset
        :param memory_depth: length of vector at each memory location
        :param config: options general to all datasets
        :param seed: random seed for generating/ordering data
        :param options: options custom to dataset
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.memory_depth = memory_depth
        self.prng = jax.random.key(seed)
        self.config = config
        self.options = options

        self.iterations = 0

        # get the cache location /Datasets/cache/
        self.common_directory = os.path.abspath(os.path.dirname(__file__) + "/..")
        self.cache_dir = os.path.join(self.common_directory, DATASETS.CACHE_LOCATION)

        # initialize the dataset if necessary
        self.initialize_dataset()

        pass

    def initialize_dataset(self):
        """To be overridden as necessary"""
        self.config.data_encoder.initialize()
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

        :param curriculum_params: inputs influencing next curriculum level. Populated with keys from Common.globals.METRICS
        """
        pass

    @abstractmethod
    def criterion(self, predictions: Array, targets: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def accuracy_metric(self, predictions: Array, targets: Array) -> Array:
        raise NotImplementedError

    def update_split(self, new_split: str):
        """Changes the split (train / test / val) used to pull data from.

        :param new_split: The split of the data to switch to
        """
        pass

    @abstractmethod
    def __next__(self) -> tuple[jax.Array, jax.Array]:
        """Gets (or creates) the next batch in the dataset."""
        raise NotImplementedError

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            CONFIG.BATCH_SIZE: self.batch_size,
            CONFIG.NUM_BATCHES: self.num_batches,
            METADATA.MEMORY_DEPTH: self.memory_depth,
            METADATA.SPLIT: self.config.split,
            METADATA.SINGLE_CURR_PER_BATCH: self.config.single_level,
            METADATA.ACCURACY_TOLERANCE: self.config.accuracy_tolerance,
            METADATA.SEED: self.seed,
        }


@dataclass
class ModelConfigInterface(ABC):
    learning_rate: float
    optimizer: Callable
    memory_M: int
    memory_N: int


class TrainingConfigInterface(ABC):
    model_config: ModelConfigInterface
    model: BackboneInterface
    model_state: TrainState
    memory_model: MemoryInterface
    _batched_run_model: Callable

    @abstractmethod
    def __init__(self, model_config) -> None:
        raise NotImplementedError

    @abstractmethod
    def run_model(
        self,
        model_params: core.FrozenDict,
        data: Array,
        output_shape: tuple[int, ...],
        memory_width: int,
    ) -> tuple[Array, tuple[Any, ...]]:
        raise NotImplementedError

    def loss_fn(self, model_params, data, target, criterion):
        output = self._batched_run_model(
            model_params, data, target.shape[1:], self.model_config.memory_N
        )
        return criterion(output, target)

    def train_step(self, data, target, criterion):
        gradient_fn = jax.value_and_grad(self.loss_fn, argnums=(0))
        ((loss), model_grads) = gradient_fn(
            self.model_state.params, data, target, criterion
        )
        self.model_state = self.model_state.apply_gradients(grads=model_grads)
        return {globals.METRICS.LOSS: loss}

    def val_step(self, data, target, criterion):
        output = self._batched_run_model(
            self.model_state.params, data, target.shape[1:], self.model_config.memory_N
        )
        return {globals.METRICS.ACCURACY: criterion(output, target)}

    @abstractmethod
    def _init_models(self) -> tuple[BackboneInterface, TrainState, MemoryInterface]:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            CONFIG.LEARNING_RATE: self.model_config.learning_rate,
            CONFIG.OPTIMIZER: self.model_config.optimizer.__name__,
            METADATA.MEMORY_DEPTH: self.model_config.memory_M,
            METADATA.MEMORY_LENGTH: self.model_config.memory_N,
        }


class DataEncoderStub(DataEncoderInterface):
    def __init__(self, memory_depth: int):
        self.memory_depth = memory_depth
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
    """Stub CurriculumScheduler which always returns a preset difficulty"""

    def __init__(self, curriculum_level: int = 0, config: dict = {}):
        self.curriculum_level: int = curriculum_level

    def update_curriculum_level(self, curriculum_params: dict):
        pass

    def get_curriculum_level(self) -> int:
        return super().get_curriculum_level()

    def get_curriculum(self, batch_size: int) -> Array:
        return jnp.repeat(self.curriculum_level, batch_size)
