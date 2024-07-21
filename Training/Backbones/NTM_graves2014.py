from dataclasses import dataclass
from typing import Callable, Type

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from Backbones.NTM_graves2014 import LSTMModel
from Common import globals
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface
from Common.TrainingInterfaces import ModelConfigInterface, TrainingConfigInterface


@dataclass
class LSTMConfig(ModelConfigInterface):
    learning_rate: float
    optimizer: Callable
    memory_class: Type[MemoryInterface]
    backbone_class: Type[LSTMModel]
    read_head_class: Type[ControllerInterface]
    write_head_class: Type[ControllerInterface]
    memory_M: int
    memory_N: int
    num_layers: int
    input_features: int


# TODO make memory width variable (set per batch based on batch's width (curriculum level))


class LSTMTrainingConfig(TrainingConfigInterface):
    model_config: LSTMConfig

    def __init__(self, model_config: LSTMConfig) -> None:
        self.model_config = model_config
        self.MEMORY_SHAPE = (
            model_config.memory_N,
            model_config.memory_M,
        )
        self.MEMORY_WIDTH = (model_config.memory_N,)
        self.model, self.model_state, self.memory_model = self._init_models()
        self._vmapped_prediction_fn = jax.vmap(
            self._prediction_fn, in_axes=(0, 0, 0, 0, 0, None, None, None)
        )

    @staticmethod
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
        # TODO make the memory size more generally settable
        self.MEMORY_WIDTH = (data.shape[1] + 1,)
        self.MEMORY_SHAPE = (
            self.MEMORY_WIDTH[0],
            model_config.memory_M,
        )

        # initial values
        read_previous = jnp.zeros((data.shape[0],) + self.MEMORY_WIDTH).at[:, 0].set(1)
        write_previous = jnp.zeros((data.shape[0],) + self.MEMORY_WIDTH).at[:, 0].set(1)
        read_data = jnp.zeros((data.shape[0], self.model_config.memory_M))
        memory_weights = jnp.zeros((data.shape[0],) + self.MEMORY_SHAPE)

        # processing loop
        for sequence in range(data.shape[1]):
            output, read_data, memory_weights, read_previous, write_previous = (
                self._vmapped_prediction_fn(
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
            ) = self._vmapped_prediction_fn(
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

    def _init_models(
        self,
    ) -> tuple[LSTMModel, train_state.TrainState, MemoryInterface]:
        rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
        key1, key2 = jax.random.split(rng_key, num=2)

        # init memory
        memory_model = self.model_config.memory_class()

        # init read and write heads
        read_head = [self.model_config.read_head_class(*self.MEMORY_SHAPE)]
        write_head = [self.model_config.write_head_class(*self.MEMORY_SHAPE)]

        # init backbone
        model = self.model_config.backbone_class(
            key1,
            self.model_config.num_layers,
            self.model_config.input_features,
            read_head,
            write_head,
            self.model_config.memory_M,
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
    from Common.Checkpoint import CheckpointWrapper, TrainingMetadata
    from Common.globals import CURRICULUM, METADATA
    from Common.TrainingInterfaces import DataloaderConfig
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import NTMMemory
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.training_loop import train

    MEMORY_DEPTH = 12
    MEMORY_WIDTH = 8
    INPUT_SIZE = 8

    model_config = LSTMConfig(
        learning_rate=1e-2,
        optimizer=optax.adamw,
        memory_class=NTMMemory,
        backbone_class=LSTMModel,
        read_head_class=NTMReadController,
        write_head_class=NTMWriteController,
        memory_M=MEMORY_DEPTH,
        memory_N=MEMORY_WIDTH,
        num_layers=1,
        input_features=INPUT_SIZE,
    )
    training_config = LSTMTrainingConfig(model_config)

    curriculum_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 2,
        CURRICULUM.CONFIGS.MAX: 10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }
    curric = CurriculumSchedulerZaremba2014(curriculum_config)
    dataset_config = DataloaderConfig(
        curriculum_scheduler=curric, accuracy_tolerance=0.1
    )
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
            f"{METADATA.COMPONENTS.CONTROLLERS.READ}{i}": head
            for i, head in enumerate(training_config.model.read_heads)
        }
        | {
            f"{METADATA.COMPONENTS.CONTROLLERS.WRITE}{i}": head
            for i, head in enumerate(training_config.model.write_heads)
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

    checkpoint_wrapper = CheckpointWrapper("NTM_graves2014_copy", delete_existing=True)

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        training_metadata=training_metadata,
        num_epochs=15,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
        checkpoint_wrapper=checkpoint_wrapper,
        # use_wandb=True,
    )

    from tqdm import tqdm

    from Visualization.MemoryWrappers import SequentialInferenceMemoryVisualizer

    training_config.memory_model = SequentialInferenceMemoryVisualizer(
        training_config.memory_model,
        save_dir="NTM_graves2014_copy",
        delete_existing=True,
        pixel_scale=64,
    )

    curriculum_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 2,
        CURRICULUM.CONFIGS.MAX: 20,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 1,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0,
    }
    curric = CurriculumSchedulerZaremba2014(curriculum_config)
    dataset_config = DataloaderConfig(curriculum_scheduler=curric)
    test_dataset = CopyLoader(
        batch_size=1,
        num_batches=10,
        memory_depth=INPUT_SIZE,
        config=dataset_config,
    )

    with tqdm(test_dataset) as pbar:
        pbar.set_description("Visualization")
        for data_batch, target_batch in pbar:
            # get a single item from the batch
            data = data_batch.at[0].get()
            target = target_batch.at[0].get()

            TEST_MEMORY_WIDTH = data.shape[0] + 1

            training_config.memory_model.set_up_inference(
                data, target, TEST_MEMORY_WIDTH, MEMORY_DEPTH
            )

            # initial state (without batch dimension)
            read_previous = jnp.zeros(TEST_MEMORY_WIDTH).at[0].set(1)
            write_previous = jnp.zeros(TEST_MEMORY_WIDTH).at[0].set(1)
            read_data = jnp.zeros(MEMORY_DEPTH)
            memory_weights = jnp.zeros((TEST_MEMORY_WIDTH, MEMORY_DEPTH))

            # processing loop
            for sequence in range(data.shape[0]):
                training_config.memory_model.update_step(
                    input_index=sequence, output_index=None
                )

                output, read_data, memory_weights, read_previous, write_previous = (
                    training_config._prediction_fn(
                        data[sequence],
                        memory_weights,
                        read_previous,
                        write_previous,
                        read_data,
                        training_config.memory_model,
                        training_config.model_state.params,
                        training_config.model_state,
                    )
                )
            for sequence in range(data.shape[0]):
                training_config.memory_model.update_step(
                    input_index=None, output_index=sequence
                )

                (
                    sequence_output,
                    read_data,
                    memory_weights,
                    read_previous,
                    write_previous,
                ) = training_config._prediction_fn(
                    jnp.zeros_like(data[sequence]),
                    memory_weights,
                    read_previous,
                    write_previous,
                    read_data,
                    training_config.memory_model,
                    training_config.model_state.params,
                    training_config.model_state,
                )

                training_config.memory_model.add_output(
                    sequence_output, sequence, memory_weights
                )

            training_config.memory_model.create_gif(loop=0, frame_duration=500)
