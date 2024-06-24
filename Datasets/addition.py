import jax

from Common import common


class AdditionLoader:
    """
    An iterable class that can be iterated over num_batches times and returns a tuple
    of arrays of integers in the (inclusive) range from min to max of size ([batch_size, 2], [batch_size, 1])
    where the first is the summands and the second is the sum. Thus it is of the form (data, label)
    """

    seed: int = common.RANDOM_SEED
    batch_size: int = common.CONFIG_BATCH_SIZE

    def __init__(
        self,
        min: int,
        max: int,
        num_batches: int,
    ):
        self.min = min
        self.max = max + 1
        self.num_batches = num_batches
        self.shape = (self.batch_size, 2)
        self.prng = jax.random.key(self.seed)
        self.iterations = 0

    def __iter__(self):
        self.iterations = 0
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self) -> tuple[jax.Array, jax.Array]:
        if self.iterations < self.num_batches:
            self.iterations += 1
            self.prng, subkey = jax.random.split(self.prng)
            summands = jax.random.randint(subkey, self.shape, self.min, self.max)

            return summands, jax.numpy.sum(summands, axis=-1)
        else:
            raise StopIteration


# basic test cases
# test by running `poetry run python -m Datasets.addition`
if __name__ == "__main__":
    num_batches = 100
    min = 0
    max = 1000000
    loader = AdditionLoader(min=min, max=max, num_batches=num_batches)
    count = 0
    for batch in loader:
        count += 1
        assert jax.numpy.all(batch[0] >= min) and jax.numpy.all(
            batch[0] <= max
        ), "Summands out of range"
        assert batch[0].shape == (
            common.CONFIG_BATCH_SIZE,
            2,
        ), "Incorrect shape of summands array"
        assert batch[1].shape == (
            common.CONFIG_BATCH_SIZE,
        ), "Incorrect shape of sum array"
        assert jax.numpy.all(
            batch[0][:, 0] + batch[0][:, 1] == batch[1]
        ), "Summands added together don't equal the sum"

    assert count == num_batches, "Incorrect number of batches generated"

    print("addition.py passed all tests")
