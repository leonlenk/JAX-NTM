import jax
import jax.numpy as jnp
from jax import Array

from Common import globals
from Common.globals import CURRICULUM, METRICS
from Common.TrainingInterfaces import CurriculumSchedulerInterface


class CurriculumSchedulerZaremba2014(CurriculumSchedulerInterface):
    def __init__(self, config: dict = {}):
        required_params = [
            CURRICULUM.CONFIGS.MIN,
            CURRICULUM.CONFIGS.MAX,
            CURRICULUM.CONFIGS.ZAREMBA2014.P1,
            CURRICULUM.CONFIGS.ZAREMBA2014.P2,
            CURRICULUM.CONFIGS.ZAREMBA2014.P3,
        ]

        assert all(
            [x in config for x in required_params]
        ), f"Missing required config params: {[x for x in required_params if x not in config]}"

        assert (
            CURRICULUM.CONFIGS.ACCURACY_THRESHOLD in config
            or CURRICULUM.CONFIGS.LOSS_THRESHOLD in config
        ), f"Either an {CURRICULUM.CONFIGS.ACCURACY_THRESHOLD} or a {CURRICULUM.CONFIGS.LOSS_THRESHOLD} is required."

        if CURRICULUM.CONFIGS.RANDOM_SEED in config:
            self.prng = jax.random.key(config[CURRICULUM.CONFIGS.RANDOM_SEED])
        else:
            self.prng = jax.random.key(globals.JAX.RANDOM_SEED)

        self.config = config

        self.curriculum_level = self.config[CURRICULUM.CONFIGS.MIN]

    def update_curriculum_level(self, curriculum_params: dict):
        check_accuracy = (
            CURRICULUM.CONFIGS.ACCURACY_THRESHOLD in self.config
            and METRICS.ACCURACY in curriculum_params
        )
        check_loss = (
            CURRICULUM.CONFIGS.LOSS_THRESHOLD in self.config
            and METRICS.LOSS in curriculum_params
        )

        assert (
            check_accuracy or check_loss
        ), "A metric matching the type of threshold set in config is required."

        # when multiple thresholds are set, adopt an "or" approach
        increment_level = False
        if check_accuracy:
            if (
                curriculum_params[METRICS.ACCURACY]
                > self.config[CURRICULUM.CONFIGS.ACCURACY_THRESHOLD]
            ):
                increment_level = True
        if check_loss:
            if (
                curriculum_params[METRICS.LOSS]
                < self.config[CURRICULUM.CONFIGS.LOSS_THRESHOLD]
            ):
                increment_level = True

        if (
            increment_level
            and self.curriculum_level < self.config[CURRICULUM.CONFIGS.MAX]
        ):
            self.curriculum_level += 1

        return

    # TODO clean up casting of randint calls
    def get_curriculum(self, batch_size: int) -> Array:
        # select between the three distributions in Zaremba2014
        self.prng, subkey = jax.random.split(self.prng)
        options = jnp.arange(1, 4)
        probs = jnp.array(
            [
                self.config[CURRICULUM.CONFIGS.ZAREMBA2014.P1],
                self.config[CURRICULUM.CONFIGS.ZAREMBA2014.P2],
                self.config[CURRICULUM.CONFIGS.ZAREMBA2014.P3],
            ]
        )
        choices = jax.random.choice(subkey, options, p=probs, shape=(batch_size,))

        # calculate "d" for each item in the batch
        ds = []
        for choice in choices:
            # option 1: uniform in ["min", "max"]
            if choice == 1:
                self.prng, subkey = jax.random.split(self.prng)
                ds.append(
                    int(
                        jax.random.randint(
                            subkey,
                            (1,),
                            self.config[CURRICULUM.CONFIGS.MIN],
                            self.config[CURRICULUM.CONFIGS.MAX] + 1,
                        )[0]
                    )
                )
                continue

            # otherwise, calculate "e"
            self.prng, subkey = jax.random.split(self.prng)
            e = jax.random.geometric(subkey, 0.5)
            # force "e" to be an int and "D" + "e" to not exceed "max"
            e = min(self.config[CURRICULUM.CONFIGS.MAX] - self.curriculum_level, int(e))

            # option 2: uniform in ["min", "D" + "e"]
            if choice == 2:
                self.prng, subkey = jax.random.split(self.prng)
                ds.append(
                    int(
                        jax.random.randint(
                            subkey,
                            (1,),
                            self.config[CURRICULUM.CONFIGS.MIN],
                            self.curriculum_level + e + 1,
                        )[0]
                    )
                )
                continue

            # option 3: "D" + "e"
            ds.append(self.curriculum_level + e)

        return jnp.array(ds)


if __name__ == "__main__":
    config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 1,
        CURRICULUM.CONFIGS.MAX: 20,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }

    batch_size = 16

    curric = CurriculumSchedulerZaremba2014(config)

    next_curriculum = curric.get_curriculum(batch_size)
    # print(f'{next_curriculum=}')
    assert next_curriculum.shape == (batch_size,)

    curriculum_params = {
        METRICS.ACCURACY: 0.95,
    }

    curric.update_curriculum_level(curriculum_params)
    assert curric.curriculum_level == 2

    curric.update_curriculum_level(curriculum_params)
    curric.update_curriculum_level(curriculum_params)
    curric.update_curriculum_level(curriculum_params)

    next_curriculum = curric.get_curriculum(batch_size)
    # print(f'{next_curriculum=}')
    assert next_curriculum.shape == (batch_size,)

    print("Passed all tests")
