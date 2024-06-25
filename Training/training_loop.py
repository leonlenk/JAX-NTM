from typing import Callable, Iterable

import jax.numpy as jnp
import wandb
from jax.typing import ArrayLike
from tqdm import tqdm

from Common import common


def average_metric(metrics: list[dict]) -> dict:
    assert len(metrics) > 0, "Must have at least one metric to aggregate"
    averaged_metric = {}
    for key in metrics[0]:
        averaged_metric[key] = jnp.average(
            jnp.array([metric[key] for metric in metrics])
        )

    return averaged_metric


def train(
    train_config: dict,
    state: ArrayLike,
    train_step: Callable[[ArrayLike, ArrayLike, dict], tuple[ArrayLike, dict]],
    train_dataset: Iterable,
    val_dataset: Iterable | None = None,
    val_step: Callable[[ArrayLike, ArrayLike, dict], dict] | None = None,
    metric_aggregator: Callable[[list[dict]], dict] = average_metric,
    wandb_tags: list[str] = [],
) -> ArrayLike:
    assert (
        common.CONFIG_EPOCHS in train_config
    ), "The number of epochs to train is a required config"

    if val_dataset is not None or val_step is not None:
        assert (
            val_dataset is not None and val_step is not None
        ), "Both val_dataset and val_step are required to perform validation"

    wandb.init(
        project=common.WANDB_PROJECT_NAME,
        job_type=common.WANDB_JOB_TRAIN,
        config=train_config,
        tags=wandb_tags,
    )

    for epoch in tqdm(range(1, train_config[common.CONFIG_EPOCHS] + 1)):
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
                f"{common.WANDB_LOG_TRAIN}_{key}": train_metric[key]
                for key in train_metric
            },
            step=epoch,
        )

        # perform validation if desired
        if val_dataset is not None and val_step is not None:
            if (
                common.CONFIG_VAL_PERIOD not in train_config
                or epoch % train_config[common.CONFIG_VAL_PERIOD] == 0
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
                        f"{common.WANDB_LOG_VAL}_{key}": val_metric[key]
                        for key in val_metric
                    },
                    step=epoch,
                )

        # TODO save best model based on validation results

    return state


# TODO add "evaluate" (aka "inference"). Add "test" separately?
