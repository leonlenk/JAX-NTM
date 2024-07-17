import jax
import jax.numpy as jnp
import optax

from Common.TrainingInterfaces import DataloaderInterface

# TODO assert memory_shape is at least 2D on init?


class CopyLoader(DataloaderInterface):
    """Returns an input and a target, both of size (batch_size, max_curriculum_level, memory_depth).
    Each item in the batch will have an array of random 0s and 1s with size (curriculum_level - 1, memory_depth - 1).
    Aside from this, all values in the array will be zero for the target.
    The input is the same as the target except for the inclusion of the delimiter (value 1) at location (curriculum_level, memory_depth).

    This is implemented by first creating the full size array and then zeroing out each section.

    Example:
        batch_size = 3
        memory_depth = 6

        data =      Array([
                [                   # first batch
                [0, 1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1]  # delimiter
                ],
                [                   # second batch
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1], # delimiter
                [0, 0, 0, 0, 0, 0]
                ],
                [                   # third batch
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1]  # delimiter
                ]], dtype=int32)

        target =    Array([
                [
                [0, 1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ],
                [
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ],
                [
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ]], dtype=int32)
    """

    def update_curriculum_level(self, curriculum_params: dict):
        self.curriculum_scheduler.update_curriculum_level(curriculum_params)

    def criterion(self, predictions, targets):
        return jnp.mean(optax.losses.l2_loss(predictions, targets))

    def accuracy_metric(self, predictions, targets):
        return jnp.mean(jnp.isclose(predictions, targets))

    def update_split(self, new_split: str):
        pass

    def __next__(self):
        if self.iterations >= self.num_batches:
            raise StopIteration
        self.iterations += 1

        # get the curriculum levels for each item in the batch
        curriculum = self.curriculum_scheduler.get_curriculum(self.batch_size)

        # create the full matrix
        self.prng, subkey = jax.random.split(self.prng)
        data = jax.random.randint(
            subkey,
            (self.batch_size, int(jnp.max(curriculum)), self.memory_depth),
            0,
            2,
        ).astype(float)

        # zero out the last memory depth
        data = data.at[:, :, -1].set(0)

        # loop through the curriculum levels and zero out the extra data columns (level - 1 columns are used)
        for i, level in enumerate(curriculum):
            data = data.at[i, level - 1 :, :].set(0)

        # this current state is the desired output
        target = data.copy()

        # loop through the curriculum levels again and add the delimiter
        for i, level in enumerate(curriculum):
            data = data.at[i, level - 1, -1].set(1)

        return data, target


if __name__ == "__main__":
    from Common.globals import CURRICULUM, DATASETS, METRICS
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014

    curric_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 1,
        CURRICULUM.CONFIGS.MAX: 20,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }

    batch_size = 5
    num_batches = 1
    memory_depth = 6

    config = {
        DATASETS.CONFIGS.CURRICULUM_SCHEDULER: CurriculumSchedulerZaremba2014(
            curric_config
        ),
    }

    copy_loader = CopyLoader(batch_size, num_batches, memory_depth, config=config)

    for data, target in copy_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    curriculum_params = {
        METRICS.ACCURACY: 0.95,
    }

    copy_loader.update_curriculum_level(curriculum_params)
    copy_loader.update_batch_params(batch_size + 1, num_batches + 1)
    for data, target in copy_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    print("Passed all tests")
