import random
import time
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state
from jax.typing import ArrayLike
from tqdm import tqdm

from Common import globals


def average_metric(metrics: list[dict]) -> dict:
    assert len(metrics) > 0, "Must have at least one metric to aggregate"
    averaged_metric = {}
    for key in metrics[0]:
        averaged_metric[key] = jnp.average(
            jnp.array([metric[key] for metric in metrics])
        )

    return averaged_metric


def train(
    project_name: str,
    train_config: dict,
    state: ArrayLike,
    train_step: Callable[[ArrayLike, ArrayLike, dict], tuple[ArrayLike, dict]],
    train_dataset: Iterable,
    val_step: Callable[[ArrayLike, ArrayLike, dict], dict] | None = None,
    val_dataset: Iterable | None = None,
    metric_aggregator: Callable[[list[dict]], dict] = average_metric,
    wandb_tags: list[str] = [],
    wandb_run_name: str | None = None,
) -> ArrayLike:
    assert (
        globals.CONFIG.EPOCHS in train_config
    ), "The number of epochs to train is a required config"

    if val_dataset is not None or val_step is not None:
        assert (
            val_dataset is not None and val_step is not None
        ), "Both val_dataset and val_step are required to perform validation"

    wandb.init(
        project=project_name,
        name=wandb_run_name,
        job_type=globals.WANDB.JOBS.TRAIN,
        config=train_config,
        tags=wandb_tags,
    )

    for epoch in tqdm(range(1, train_config[globals.CONFIG.EPOCHS] + 1)):
        # train the model on each batch
        train_metrics: list[dict] = []
        for data in tqdm(train_dataset):
            # run the train step function
            state, metrics = train_step(state, data, train_config)
            # record the results
            train_metrics.append(metrics)

        # combine the metrics from each batch into a single dictionary to log
        train_metric = metric_aggregator(train_metrics)
        wandb.log(
            {
                f"{globals.WANDB.LOGS.TRAIN}_{key}": train_metric[key]
                for key in train_metric
            },
            step=epoch,
        )

        # perform validation if desired
        if val_dataset is not None and val_step is not None:
            if (
                globals.CONFIG.VAL_PERIOD not in train_config
                or epoch % train_config[globals.CONFIG.VAL_PERIOD] == 0
            ):
                val_metrics: list[dict] = []
                for data in tqdm(train_dataset):
                    # run the val step function
                    metrics = val_step(state, data, train_config)
                    # record the results
                    val_metrics.append(metrics)
                # combine the metrics from each batch into a single dictionary to log
                val_metric = metric_aggregator(val_metrics)
                wandb.log(
                    {
                        f"{globals.WANDB.LOGS.VAL}_{key}": val_metric[key]
                        for key in val_metric
                    },
                    step=epoch,
                )

        # TODO save best model based on validation results

    return state


# TODO add "evaluate" (aka "inference"). Add "test" separately?


def init_models(train_config, model_config):
    MEMORY_SHAPE = (
        train_config[globals.MACHINES.GRAVES2014.MEMORY.N],
        train_config[globals.MACHINES.GRAVES2014.MEMORY.M],
    )
    MEMORY_WIDTH = (1, train_config[globals.MACHINES.GRAVES2014.MEMORY.N])
    DATA_SHAPE = (1, train_config[globals.MACHINES.GRAVES2014.MEMORY.M])

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2, key3, key4 = jax.random.split(rng_key, num=4)

    # init memory
    memory_model = model_config[globals.MODELS.MEMORY](
        key1,
        (1, *MEMORY_SHAPE),
        model_config[globals.MODELS.OPTIMIZER](
            train_config[globals.CONFIG.LEARNING_RATE]
        ),
    )

    # init read controller
    read_controller = model_config[globals.MODELS.READ_CONTROLLER](*MEMORY_SHAPE)
    read_variables = read_controller.init(
        key2,
        jnp.ones(DATA_SHAPE),
        jnp.ones(MEMORY_WIDTH),
        memory_model.weights,
        memory_model,
    )
    read_state = train_state.TrainState.create(
        apply_fn=read_controller.apply,
        tx=model_config[globals.MODELS.OPTIMIZER],
        params=read_variables[globals.JAX.PARAMS],
    )

    # init write controller
    write_controller = model_config[globals.MODELS.WRITE_CONTROLLER](*MEMORY_SHAPE)
    write_variables = write_controller.init(
        key3,
        jnp.ones(DATA_SHAPE),
        jnp.ones(MEMORY_WIDTH),
        memory_model.weights,
        memory_model,
    )
    write_state = train_state.TrainState.create(
        apply_fn=write_controller.apply,
        tx=model_config[globals.MODELS.OPTIMIZER],
        params=write_variables[globals.JAX.PARAMS],
    )

    return read_state, write_state, memory_model


if __name__ == "__main__":
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Memories.NTM_graves2014 import Memory

    train_config = {
        globals.CONFIG.EPOCHS: 5,
        globals.CONFIG.BATCH_SIZE: 32,
        globals.CONFIG.LEARNING_RATE: 1e-4,
        globals.MACHINES.GRAVES2014.MEMORY.N: 8,
        globals.MACHINES.GRAVES2014.MEMORY.M: 12,
    }

    model_config = {
        globals.MODELS.MEMORY: Memory,
        globals.MODELS.WRITE_CONTROLLER: NTMWriteController,
        globals.MODELS.READ_CONTROLLER: NTMReadController,
        globals.MODELS.OPTIMIZER: optax.adam,
    }

    state = 1
    dataset = range(3)

    def train_step(state, data, train_config):
        time.sleep(0.3)
        metrics = {}
        metrics[globals.METRICS.LOSS] = random.uniform(0, 1 / state)
        metrics[globals.METRICS.ACCURACY] = random.uniform(0, 1 - 1 / state)
        return state + 1, metrics

    output_state = train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        train_config=train_config,
        state=state,
        train_step=train_step,
        train_dataset=dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
    )
