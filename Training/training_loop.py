from typing import Callable, Iterable

import jax.numpy as jnp
from jax.typing import ArrayLike
from tqdm import tqdm

import wandb
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


if __name__ == "__main__":
    import random
    import time

    train_config = {
        globals.CONFIG.EPOCHS: 5,
        globals.CONFIG.LEARNING_RATE: 1e-4,
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
