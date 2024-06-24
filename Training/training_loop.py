from typing import Callable, Iterable

import wandb
from jax.typing import ArrayLike
from tqdm import tqdm

from Common import common


def train(
    train_config: dict,
    state: ArrayLike,
    train_step: Callable[[ArrayLike, ArrayLike, dict], tuple[ArrayLike, ArrayLike]],
    train_dataset: Iterable,
    val_dataset: Iterable | None = None,
    wandb_tags: list[str] = [],
) -> ArrayLike:
    assert (
        common.CONFIG_EPOCHS in train_config
    ), "The number of epochs to train is a required config"

    wandb.init(
        project=common.WANDB_PROJECT_NAME,
        job_type=common.WANDB_JOB_TRAIN,
        config=train_config,
        tags=wandb_tags,
    )

    for epoch in tqdm(range(1, train_config[common.CONFIG_EPOCHS] + 1)):
        # TODO add function to convert list of metrics (loss, accuracy, etc) into a loggable piece of information
        train_metrics = []
        for data in tqdm(train_dataset):
            state, metrics = train_step(state, data, train_config)
            train_metrics.append(metrics)

        # TODO add validation loop. Check if val_dataset is none. Add config option for how often to do validation?

        # TODO save best model based on validation results (use single-valued metric function?)

        # TODO add wandb.log
        # wandb.log({
        #     "Train Loss": loss,
        # }, step=epoch)

    return state


# TODO add "evaluate" (aka "inference"). Add "test" separately?
