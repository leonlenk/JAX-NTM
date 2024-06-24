from typing import Callable, Iterable

import wandb
from jax.typing import ArrayLike
from tqdm import tqdm

from Common import common


def train(
    wandb_conf: dict,
    state: ArrayLike,
    train_step: Callable,
    train_dataset: Iterable,
    val_dataset: Iterable | None = None,
    wandb_tags: list[str] = [],
):
    assert common.CONFIG_EPOCHS in wandb_conf, "Number of epochs required"

    wandb.init(
        project=common.WANDB_PROJECT_NAME,
        job_type=common.WANDB_JOB_TRAIN,
        config=wandb_conf,
        tags=wandb_tags,
    )

    for epoch in tqdm(range(1, wandb_conf[common.CONFIG_EPOCHS] + 1)):
        # TODO add function to convert list of metrics (loss, accuracy, etc) into a loggable piece of information
        train_metrics = []
        for data in tqdm(train_dataset):
            state, metrics = train_step(state, data)
            train_metrics.append(metrics)

        # TODO add validation loop. Check if val_dataset is none. Add config option for how often to do validation?

        # TODO save best model based on validation results (use single-valued metric function?)

        # TODO add wandb.log
        # wandb.log({
        #     "Train Loss": loss,
        # }, step=epoch)


# TODO add "evaluate" (aka "inference"). Add "test" separately?
