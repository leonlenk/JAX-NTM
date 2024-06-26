from abc import ABC, abstractmethod

import jax
from jax import Array

from Common import globals


class CurriculumSchedulerInterface(ABC):
    @abstractmethod
    def __init__(self, config: dict = {}):
        """Initializes the curriculum scheduler.

        :param config: a set of configuration options custom to the type of scheduler. Populated with keys from Common.globals.CURRICULUM.OPTIONS
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
        curriculum_scheduler: CurriculumSchedulerInterface,
        seed: int = globals.JAX.RANDOM_SEED,
    ):
        """Initializes the dataloader.

        :param batch_size: number of samples per batch
        :param num_batches: number of batches in the dataset
        :param curriculum_scheduler: object which informs the dataloader what difficulty samples to prepare
        :param seed: random seed for generating/ordering data
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.curriculum_scheduler = curriculum_scheduler
        self.prng = jax.random.key(seed)
        self.iterations = 0
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
    def __next__(self) -> tuple[jax.Array, jax.Array]:
        """Gets (or creates) the next batch in the dataset."""
        pass
