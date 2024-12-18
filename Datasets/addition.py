import jax
import jax.numpy as jnp
import optax

from Common.globals import DATASETS
from Common.TrainingInterfaces import DataloaderInterface

# TODO
# change the randint stuff to jax.random.choice with an array of difficulties and an array of probabilities for each
# create a new common class for holding those two arrays?
# make it auto-generate the arrays given a min/max or given a min/max + target + weights or something?
# add a function to update the range


class BinaryAdditionLoader(DataloaderInterface):
    """Returns an input and a target
        The input is of size (batch_size, memory_depth, 2 * (curriculum_level + 1)).
        The output is of size (batch_size, memory_depth, curriculum_level + 1).
    The addition problem will be composed of two numbers (the augend and the addend)
        These are randomly selected from [0,2^curriculum_level)
    The first element of each memory location will be 1s and 0s representing the number in binary.
        these will be left-aligned and have the least significant bit come first
    There will be two extra memory locations as delimiters, one after each number.
        The first 3 values in their memory locations will be (0,1,0,...) and (0,0,1,...) respectively.
    Aside from this, all values in the array will be zero.

    The target will be of length (2 * curriculum_level) + 1.
        The sum of the two numbers will be encoded in the same way.

    Example:
        batch_size = 2
        memory_depth = 6

        augends =   Array([3, 0], dtype=int32)
        addends =   Array([0, 1], dtype=int32)
        sums =      Array([3, 1], dtype=int32)

        data =      Array([
                [                           # first batch
                [1., 0., 0., 0., 0., 0.],   # 1s place
                [1., 0., 0., 0., 0., 0.],   # 2s place
                [0., 1., 0., 0., 0., 0.],   # first delimiter -> augend = 3
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.]    # second delimiter -> addend = 0
                ],
                [
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.]
                ]], dtype=float32)

        target =    Array([
                [                           # first batch
                [1., 0., 0., 0., 0., 0.],   # 1s place
                [1., 0., 0., 0., 0., 0.],   # 2s place
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]
                ],                          # sum = 3
                [                           # second batch
                [1., 0., 0., 0., 0., 0.],   # 1s place
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]
                ]], dtype=float32)          # sum = 1
    """

    def update_curriculum_level(self, curriculum_params: dict):
        self.config.curriculum_scheduler.update_curriculum_level(curriculum_params)

    # TODO is this the right criterion to use?
    def criterion(self, predictions, targets):
        truncated_predictions = predictions.at[1, :].get()
        truncated_targets = targets.at[1, :].get()
        return jnp.mean(
            optax.losses.l2_loss(
                jax.nn.sigmoid(truncated_predictions), truncated_targets
            )
        )
        # return jnp.mean(jnp.abs(predictions - targets))  # mean absolute error (l1) loss

    def accuracy_metric(self, predictions, targets):
        truncated_predictions = predictions.at[1, :].get()
        truncated_targets = targets.at[1, :].get()
        return jnp.mean(
            jnp.isclose(
                jax.nn.sigmoid(truncated_predictions),
                truncated_targets,
                atol=self.config.accuracy_tolerance,
            )
        )

    def update_split(self, new_split: str):
        pass

    def __next__(self):
        if self.iterations >= self.num_batches:
            raise StopIteration
        self.iterations += 1

        self.interleaved: bool = self.options.get(
            DATASETS.ADD.CONFIGS.INTERLEAVED, True
        )

        # get the curriculum levels for each item in the batch
        if self.config.single_level:
            curriculum = jnp.repeat(
                self.config.curriculum_scheduler.get_curriculum(1), self.batch_size
            )
        else:
            curriculum = self.config.curriculum_scheduler.get_curriculum(
                self.batch_size
            )

        # create the full size matrix of zeros for the data and target
        max_curriculum_level = jnp.max(curriculum)

        if self.interleaved:
            data = jnp.zeros(
                (self.batch_size, 2 * max_curriculum_level + 1, self.memory_depth)
            )
        else:
            data = jnp.zeros(
                (self.batch_size, 2 * (max_curriculum_level + 1), self.memory_depth)
            )
        target = jnp.zeros(
            (self.batch_size, max_curriculum_level + 1, self.memory_depth)
        )

        # get the augends, addends, and sums
        self.prng, subkey = jax.random.split(self.prng)
        augends = jax.random.randint(subkey, (self.batch_size,), 0, 2**curriculum)
        self.prng, subkey = jax.random.split(self.prng)
        addends = jax.random.randint(subkey, (self.batch_size,), 0, 2**curriculum)
        sums = augends + addends

        # loop through the batches and fill in the array
        for i in range(self.batch_size):
            curriculum_level = curriculum[i]
            augend = augends[i]
            addend = addends[i]
            sum = sums[i]

            # set the delimiters
            if self.interleaved:
                data = data.at[i, 2 * curriculum_level, 1].set(1)
            else:
                data = data.at[i, curriculum_level, 1].set(1)
                data = data.at[i, 2 * curriculum_level + 1, 2].set(1)

            # fill in the augend and addend
            for j in range(curriculum_level):
                if self.get_bit_bool(augend, j):
                    if self.interleaved:
                        data = data.at[i, j * 2, 0].set(1)
                    else:
                        data = data.at[i, j, 0].set(1)

                if self.get_bit_bool(addend, j):
                    if self.interleaved:
                        data = data.at[i, j * 2 + 1, 0].set(1)
                    else:
                        data = data.at[i, curriculum_level + 1 + j, 0].set(1)

            # fill in the target
            target_length = target.shape[1]
            for j in range(target_length):
                if self.get_bit_bool(sum, j):
                    target = target.at[i, j, 0].set(1)

        return data, target

    def get_bit_bool(self, value, n):
        return (value >> n & 1) != 0


if __name__ == "__main__":
    from Common.globals import CURRICULUM, METRICS
    from Common.TrainingInterfaces import DataloaderConfig
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014

    curric_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 1,
        CURRICULUM.CONFIGS.MAX: 10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }

    batch_size = 2
    num_batches = 1
    memory_depth = 6

    config = DataloaderConfig(
        curriculum_scheduler=CurriculumSchedulerZaremba2014(curric_config)
    )
    add_loader = BinaryAdditionLoader(
        batch_size, num_batches, memory_depth, config=config
    )

    for data, target in add_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    curriculum_params = {
        METRICS.ACCURACY: 0.95,
    }

    add_loader.update_curriculum_level(curriculum_params)
    add_loader.update_batch_params(batch_size + 1, num_batches + 1)
    for data, target in add_loader:
        # print(f'{data=}')
        # print(f'{target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3

    print("Passed all tests")
