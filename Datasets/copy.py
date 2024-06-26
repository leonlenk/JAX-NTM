import jax
import jax.numpy as jnp

from Common import globals
from Common.TrainingInterfaces import DataloaderInterface

# TODO assert memory_shape is at least 2D on init?


class CopyLoader(DataloaderInterface):
    def update_curriculum_level(self, curriculum_params: dict):
        self.curriculum_scheduler.update_curriculum_level(curriculum_params)

    def __next__(self):
        """For each item, create an array of random 0s and 1s with size
        (memory_depth - 1, curriculum_level - 1)
        then add a final delimiter column with all 0s except for a 1 at (memory_depth, curriculum_level)
        then pad with zeros until the item is of size (memory_depth, max_curriculum_level)
        finally, combine all items together into an array of size (batch_size, memory_depth, max_curriculum_level)

        The expected output is the same array, but with the 1 in the delimiter column equal to zero.

        This is actually done by creating the full size array and then zeroing out each section.
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
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014

    config = {
        globals.CURRICULUM.OPTIONS.ACCURACY_THRESHOLD: 0.9,
        globals.CURRICULUM.OPTIONS.MIN: 1,
        globals.CURRICULUM.OPTIONS.MAX: 20,
        globals.CURRICULUM.OPTIONS.ZAREMBA2014.P1: 0.10,
        globals.CURRICULUM.OPTIONS.ZAREMBA2014.P2: 0.25,
        globals.CURRICULUM.OPTIONS.ZAREMBA2014.P3: 0.65,
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
        globals.CURRICULUM.PARAMS.ACCURACY: 0.95,
    }

    copy_loader.update_curriculum_level(curriculum_params)
    copy_loader.update_batch_params(batch_size + 1, num_batches + 1)
    for data, target in copy_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    print("Passed all tests")
