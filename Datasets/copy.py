import jax
import jax.numpy as jnp

from Common.TrainingInterfaces import DataloaderInterface

# TODO assert memory_shape is at least 2D on init?


class CopyLoader(DataloaderInterface):
    def update_curriculum_level(self, curriculum_params: dict):
        self.curriculum_scheduler.update_curriculum_level(curriculum_params)

    def __next__(self):
        """Returns an input and a target, both of size (batch_size, memory_depth, max_curriculum_level).
        Each item in the batch will have an array of random 0s and 1s with size (memory_depth - 1, curriculum_level - 1).
        Aside from this, all values in the array will be zero for the target.
        The input is the same as the target except for the inclusion of the delimiter (value 1) at location (memory_depth, curriculum_level).

        This is implemented by first creating the full size array and then zeroing out each section.
        """
        if self.iterations >= self.num_batches:
            raise StopIteration
        self.iterations += 1

        # get the curriculum levels for each item in the batch
        curriculum = self.curriculum_scheduler.get_curriculum(self.batch_size)

        # create the full matrix
        self.prng, subkey = jax.random.split(self.prng)
        data = jax.random.randint(
            subkey,
            (self.batch_size, self.memory_shape[1], int(jnp.max(curriculum))),
            0,
            2,
        )

        # zero out the last memory depth
        data = data.at[:, -1, :].set(0)

        # loop through the curriculum levels and zero out the extra data columns (level - 1 columns are used)
        for i, level in enumerate(curriculum):
            data = data.at[i, :, level - 1 :].set(0)

        # this current state the desired output
        target = data.copy()

        # loop through the curriculum levels again and add the delimiter
        for i, level in enumerate(curriculum):
            data = data.at[i, -1, level - 1].set(1)

        return data, target


if __name__ == "__main__":
    from Common.globals import CURRICULUM
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014

    config = {
        CURRICULUM.OPTIONS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.OPTIONS.MIN: 1,
        CURRICULUM.OPTIONS.MAX: 20,
        CURRICULUM.OPTIONS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.OPTIONS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.OPTIONS.ZAREMBA2014.P3: 0.65,
    }

    batch_size = 7
    num_batches = 1
    memory_shape = (5, 6)

    curric = CurriculumSchedulerZaremba2014(config)

    copy_loader = CopyLoader(batch_size, num_batches, memory_shape, curric)

    for data, target in copy_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    curriculum_params = {
        CURRICULUM.PARAMS.ACCURACY: 0.95,
    }

    copy_loader.update_curriculum_level(curriculum_params)
    copy_loader.update_batch_params(batch_size + 1, num_batches + 1)
    for data, target in copy_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    print("Passed all tests")
