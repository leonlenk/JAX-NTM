from typing import Callable

import jax.numpy as jnp
from tqdm import tqdm

import wandb
from Common.Checkpoint import CheckpointWrapper, TrainingMetadata
from Common.globals import CHECKPOINTS, CONFIG, METRICS, WANDB
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
    training_metadata: TrainingMetadata,
    num_epochs: int,
    train_dataset: DataloaderInterface,
    val_period: int = 1,
    val_dataset: DataloaderInterface | None = None,
    metric_aggregator: Callable[[list[dict]], dict] = average_metric,
    checkpoint_wrapper: CheckpointWrapper | None = None,
    current_epoch: int = 1,
    use_wandb: bool = False,
    wandb_tags: list[str] = [],
    wandb_run_name: str | None = None,
):
    if use_wandb:
        wandb_config = {
            CONFIG.EPOCHS: num_epochs,
            CHECKPOINTS.METADATA: training_metadata.get_metadata(),
        }

        wandb.init(
            project=project_name,
            name=wandb_run_name,
            job_type=WANDB.JOBS.TRAIN,
            config=wandb_config,
            tags=wandb_tags,
        )

    for epoch in range(current_epoch, num_epochs + 1):
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
                pbar.set_postfix(loss=f"{metrics[METRICS.LOSS]:.4f}")

        # combine the metrics from each batch into a single dictionary to log
        train_metric = metric_aggregator(train_metrics)
        if use_wandb:
            wandb_log_dict = {
                f"{WANDB.LOGS.TRAIN}_{key}": train_metric[key] for key in train_metric
            }
            wandb_log_dict |= {
                METRICS.CURRICULUM_LEVEL: train_dataset.config.curriculum_scheduler.get_curriculum_level()
            }
            wandb.log(
                wandb_log_dict,
                step=epoch,
            )

        # perform validation if desired
        if val_dataset is not None:
            if epoch % val_period == 0:
                val_metrics: list[dict] = []
                with tqdm(val_dataset) as pbar:
                    pbar.set_description("Val Step")
                    for data, target in pbar:
                        # run the val step function
                        metrics = training_config.val_step(
                            data, target, val_dataset.accuracy_metric
                        )
                        # record the results
                        val_metrics.append(metrics)
                        pbar.set_postfix(acc=f"{metrics[METRICS.ACCURACY] * 100:.2f}%")

                # combine the metrics from each batch into a single dictionary to log
                val_metric = metric_aggregator(val_metrics)
                if use_wandb:
                    wandb_log_dict = {
                        f"{WANDB.LOGS.VAL}_{key}": val_metric[key] for key in val_metric
                    }
                    wandb_log_dict |= {
                        METRICS.CURRICULUM_LEVEL: val_dataset.config.curriculum_scheduler.get_curriculum_level()
                    }
                    wandb.log(
                        wandb_log_dict,
                        step=epoch,
                    )

                # update curriculum level using val metrics if available
                train_dataset.update_curriculum_level(val_metric)
                val_dataset.update_curriculum_level(val_metric)

            # if no validation is being performed, update curriculum level using train metrics
            else:
                train_dataset.update_curriculum_level(train_metric)

        # TODO add metrics to checkpointer to save best model
        if checkpoint_wrapper is not None:
            checkpoint_wrapper.save_checkpoint(
                training_config.model_state, training_metadata.get_metadata()
            )
