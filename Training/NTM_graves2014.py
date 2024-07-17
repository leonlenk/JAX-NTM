from dataclasses import dataclass
from functools import partial
from typing import Callable, Type

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from Common import globals
from Common.BackboneInterface import BackboneInterface
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface
from Common.TrainingInterfaces import ModelConfigInterface, TrainingConfigInterface


@dataclass
class ModelConfig(ModelConfigInterface):
    learning_rate: float
    optimizer: Callable
    memory_class: Type[MemoryInterface]
    backbone_class: Type[BackboneInterface]
    read_head_class: Type[ControllerInterface]
    write_head_class: Type[ControllerInterface]
    memory_M: int
    memory_N: int
    num_layers: int
    input_features: int


class TrainingConfig(TrainingConfigInterface):
    model_config: ModelConfig

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.MEMORY_SHAPE = (
            model_config.memory_N,
            model_config.memory_M,
        )
        self.MEMORY_WIDTH = (model_config.memory_N,)
        self.model, self.model_state, self.memory_model = self._init_models()

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None, None))
    def _prediction_fn(
        data,
        memory_weights,
        read_previous,
        write_previous,
        read_data,
        memory_model,
        model_params,
        model_state,
    ):
        data = jnp.concat((data, read_data), axis=-1)
        (
            (output, read_data, memory_weights, read_previous, write_previous),
            variables,
        ) = model_state.apply_fn(
            {globals.JAX.PARAMS: model_params},
            data,
            memory_weights,
            read_previous,
            write_previous,
            memory_model,
            mutable=["state"],
        )
        return output, read_data, memory_weights, read_previous, write_previous

    def _run_model(self, model_params, data, output_shape):
        # initial values
        read_previous = jnp.zeros((data.shape[0],) + self.MEMORY_WIDTH).at[:, 0].set(1)
        write_previous = jnp.zeros((data.shape[0],) + self.MEMORY_WIDTH).at[:, 0].set(1)
        read_data = jnp.zeros((data.shape[0], self.model_config.memory_M))
        memory_weights = jnp.zeros((data.shape[0],) + self.MEMORY_SHAPE)

        # processing loop
        for sequence in range(data.shape[1]):
            output, read_data, memory_weights, read_previous, write_previous = (
                self._prediction_fn(
                    data[:, sequence],
                    memory_weights,
                    read_previous,
                    write_previous,
                    read_data,
                    self.memory_model,
                    model_params,
                    self.model_state,
                )
            )

        output = jnp.empty(output_shape)
        for sequence in range(output_shape[1]):
            (
                sequence_output,
                read_data,
                memory_weights,
                read_previous,
                write_previous,
            ) = self._prediction_fn(
                jnp.zeros_like(data[:, 0]),
                memory_weights,
                read_previous,
                write_previous,
                read_data,
                self.memory_model,
                model_params,
                self.model_state,
            )
            output = output.at[:, sequence].set(sequence_output)
        return output, ("placeholder",)

    def loss_fn(self, model_params, data, target, criterion):
        output, metrics = self._run_model(model_params, data, target.shape)
        return criterion(output, target), metrics

    def train_step(self, data, target, criterion):
        gradient_fn = jax.value_and_grad(self.loss_fn, argnums=(0), has_aux=True)
        ((loss, (metric,)), model_grads) = gradient_fn(
            self.model_state.params, data, target, criterion
        )
        self.model_state = self.model_state.apply_gradients(grads=model_grads)
        return {globals.METRICS.LOSS: loss}

    def val_step(self, data, target, criterion):
        output, (metric,) = self._run_model(self.model_state.params, data, target.shape)
        return {globals.METRICS.ACCURACY: criterion(output, target)}

    def _init_models(
        self,
    ) -> tuple[BackboneInterface, train_state.TrainState, MemoryInterface]:
        rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
        key1, key2 = jax.random.split(rng_key, num=2)

        # init memory
        memory_model = self.model_config.memory_class()

        # init read and write heads
        read_head = self.model_config.read_head_class(*self.MEMORY_SHAPE)
        write_head = self.model_config.write_head_class(*self.MEMORY_SHAPE)

        # init backbone
        model = self.model_config.backbone_class(
            key1,
            self.model_config.memory_M,
            self.model_config.num_layers,
            self.model_config.input_features,
            read_head,
            write_head,
        )
        init_input = jnp.ones(
            (self.model_config.input_features + self.model_config.memory_M)
        )
        memory_weights = jnp.ones(self.MEMORY_SHAPE)
        params = model.init(
            key1,
            init_input,
            memory_weights,
            jnp.ones(self.MEMORY_WIDTH),
            jnp.ones(self.MEMORY_WIDTH),
            memory_model,
        )
        model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            tx=optax.adam(self.model_config.learning_rate),
            params=params[globals.JAX.PARAMS],
        )

        return model, model_state, memory_model


if __name__ == "__main__":
    from Backbones.NTM_graves2014 import LSTMModel
    from Common.Checkpoint import CheckpointWrapper, TrainingMetadata
    from Common.globals import CURRICULUM, METADATA
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import NTMMemory
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.training_loop import train

    MEMORY_DEPTH = 12
    INPUT_SIZE = 8

    model_config = ModelConfig(
        learning_rate=1e-2,
        optimizer=optax.adamw,
        memory_class=NTMMemory,
        backbone_class=LSTMModel,
        read_head_class=NTMReadController,
        write_head_class=NTMWriteController,
        memory_M=MEMORY_DEPTH,
        memory_N=8,
        num_layers=1,
        input_features=INPUT_SIZE,
    )
    training_config = TrainingConfig(model_config)

    curriculum_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 3,
        CURRICULUM.CONFIGS.MAX: 3,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }
    curric = CurriculumSchedulerZaremba2014(curriculum_config)
    dataset_config = {globals.DATASETS.CONFIGS.CURRICULUM_SCHEDULER: curric}
    train_dataset = CopyLoader(
        batch_size=256,
        num_batches=50,
        memory_depth=INPUT_SIZE,
        config=dataset_config,
    )

    val_dataset = CopyLoader(
        batch_size=256,
        num_batches=5,
        memory_depth=INPUT_SIZE,
        config=dataset_config,
    )

    training_metadata = TrainingMetadata(
        Backbone={str(0): training_config.model},
        Controller={
            f"{METADATA.COMPONENTS.CONTROLLERS.READ}{0}": training_config.model.read_head,
            f"{METADATA.COMPONENTS.CONTROLLERS.WRITE}{0}": training_config.model.write_head,
        },
        Memory={str(0): training_config.memory_model},
        DataEncoder={},
        CurriculumScheduler={str(0): curric},
        Dataloader={
            METADATA.COMPONENTS.DATALOADERS.TRAIN: train_dataset,
            METADATA.COMPONENTS.DATALOADERS.VAL: val_dataset,
        },
        TrainingConfig={str(0): training_config},
    )

    checkpoint_wrapper = CheckpointWrapper(
        "NTM_graves2014_copy_test", delete_existing=True
    )

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        num_epochs=5,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
        checkpoint_wrapper=checkpoint_wrapper,
        training_metadata=training_metadata,
    )
