from typing import Callable, Iterable

import jax.numpy as jnp
import optax
from jax.typing import ArrayLike
from tqdm import tqdm

import wandb
from Common import globals
from Common.TrainingInterfaces import DataloaderInterface, TrainingConfigInterface


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
    training_config: TrainingConfigInterface,
    num_epochs: int,
    train_dataset: DataloaderInterface,
    val_period: int | None = None,
    val_step: Callable[[ArrayLike, ArrayLike, dict], dict] | None = None,
    val_dataset: Iterable | None = None,
    metric_aggregator: Callable[[list[dict]], dict] = average_metric,
    use_wandb: bool = False,
    wandb_tags: list[str] = [],
    wandb_run_name: str | None = None,
):
    if val_dataset is not None or val_step is not None:
        assert (
            val_dataset is not None and val_step is not None
        ), "Both val_dataset and val_step are required to perform validation"

    if use_wandb:
        wandb_config = {
            globals.CONFIG.EPOCHS: num_epochs,
            globals.CONFIG.LEARNING_RATE: training_config.model_config.learning_rate,
            globals.MACHINES.GRAVES2014.MEMORY.N: training_config.model_config.memory_N,
            globals.MACHINES.GRAVES2014.MEMORY.M: training_config.model_config.memory_M,
        }

        wandb.init(
            project=project_name,
            name=wandb_run_name,
            job_type=globals.WANDB.JOBS.TRAIN,
            config=wandb_config,
            tags=wandb_tags,
        )

    for epoch in range(1, num_epochs + 1):
        # train the model on each batch
        train_metrics: list[dict] = []
        with tqdm(train_dataset) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for data, target in pbar:
                # run the train step function
                metrics = training_config.train_step(
                    data, target, train_dataset.criterion
                )
                # record the results
                train_metrics.append(metrics)
                pbar.set_postfix(loss=metrics[globals.METRICS.LOSS])

        # combine the metrics from each batch into a single dictionary to log
        train_metric = metric_aggregator(train_metrics)
        if use_wandb:
            wandb.log(
                {
                    f"{globals.WANDB.LOGS.TRAIN}_{key}": train_metric[key]
                    for key in train_metric
                },
                step=epoch,
            )

        # perform validation if desired
        if val_dataset is not None and val_step is not None:
            if val_period is not None and epoch % val_period == 0:
                val_metrics: list[dict] = []
                for data, target in tqdm(train_dataset):
                    # run the val step function
                    metrics = training_config.val_step(data)
                    # record the results
                    val_metrics.append(metrics)
                # combine the metrics from each batch into a single dictionary to log
                val_metric = metric_aggregator(val_metrics)
                if use_wandb:
                    wandb.log(
                        {
                            f"{globals.WANDB.LOGS.VAL}_{key}": val_metric[key]
                            for key in val_metric
                        },
                        step=epoch,
                    )

        # TODO save best model based on validation results


# TODO add "evaluate" (aka "inference"). Add "test" separately?
if __name__ == "__main__":
    from Backbone.NTM_graves2014 import LSTMModel
    from Common.globals import CURRICULUM
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import Memory
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.NTM_graves2014 import ModelConfig, TrainingConfig

    MEMORY_DEPTH = 12

    model_config = ModelConfig(
        learning_rate=1e-3,
        optimizer=optax.adam,
        memory_class=Memory,
        backbone_class=LSTMModel,
        read_head_class=NTMReadController,
        write_head_class=NTMWriteController,
        memory_M=MEMORY_DEPTH,
        memory_N=8,
        num_layers=3,
        num_outputs=12,
        input_length=1,
        input_features=12,
    )
    training_config = TrainingConfig(model_config)

    curriculum_config = {
        CURRICULUM.OPTIONS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.OPTIONS.MIN: 1,
        CURRICULUM.OPTIONS.MAX: 10,
        CURRICULUM.OPTIONS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.OPTIONS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.OPTIONS.ZAREMBA2014.P3: 0.65,
    }
    curric = CurriculumSchedulerZaremba2014(curriculum_config)
    dataset = CopyLoader(
        batch_size=32,
        num_batches=10,
        memory_depth=MEMORY_DEPTH,
        curriculum_scheduler=curric,
    )

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        num_epochs=5,
        train_dataset=dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
    )
