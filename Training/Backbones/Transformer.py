from dataclasses import dataclass
from functools import partial
from typing import Callable, Type

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from Backbones.Transformer import TransformerModel
from Common import globals
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface
from Common.TrainingInterfaces import ModelConfigInterface, TrainingConfigInterface


@dataclass
class TransformerConfig(ModelConfigInterface):
    prng_key: jax.Array
    learning_rate: float
    optimizer: Callable
    memory_class: Type[MemoryInterface]
    backbone_class: Type[TransformerModel]
    read_head_class: Type[ControllerInterface]
    write_head_class: Type[ControllerInterface]
    memory_M: int
    memory_N: int
    num_layers: int
    dim_model: int
    num_heads: int
    dim_ff: int
    max_sequence_len: int


class TransformerTrainingConfig(TrainingConfigInterface):
    model_config: TransformerConfig

    def __init__(self, model_config: TransformerConfig) -> None:
        self.model_config = model_config
        self.MEMORY_SHAPE = (
            model_config.num_layers,
            model_config.memory_N,
            model_config.memory_M,
        )
        self.SINGLE_MEMORY_SHAPE = (
            model_config.memory_N,
            model_config.memory_M,
        )
        self.CONTROLLER_PREVIOUS_SHAPE = (
            model_config.num_layers,
            model_config.memory_N,
        )
        self.model, self.model_state, self.memory_model = self._init_models()

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0, 0, 0, None, None, None))
    def _prediction_fn(
        data,
        memory_weights,
        read_previous,
        write_previous,
        memory_model,
        model_params,
        model_state,
    ):
        (
            output,
            memory_weights,
            read_previous,
            write_previous,
        ) = model_state.apply_fn(
            {globals.JAX.PARAMS: model_params},
            data,
            memory_weights,
            read_previous,
            write_previous,
            memory_model,
        )
        return output, memory_weights, read_previous, write_previous

    def _run_model(self, model_params, data, output_shape):
        # initial values
        read_previous = (
            jnp.zeros((data.shape[0],) + self.CONTROLLER_PREVIOUS_SHAPE)
            .at[:, :, 0]
            .set(1)
        )
        write_previous = (
            jnp.zeros((data.shape[0],) + self.CONTROLLER_PREVIOUS_SHAPE)
            .at[:, :, 0]
            .set(1)
        )
        memory_weights = jnp.zeros((data.shape[0],) + self.MEMORY_SHAPE)

        # processing loop
        for sequence in range(1, data.shape[1] + 1):
            output, memory_weights, read_previous, write_previous = self._prediction_fn(
                data[:, :sequence],
                memory_weights,
                read_previous,
                write_previous,
                self.memory_model,
                model_params,
                self.model_state,
            )

        # prediciton loop
        output = jnp.empty(output_shape)
        for sequence in range(output_shape[1] + 1):
            (
                sequence_output,
                memory_weights,
                read_previous,
                write_previous,
            ) = self._prediction_fn(
                data,
                memory_weights,
                read_previous,
                write_previous,
                self.memory_model,
                model_params,
                self.model_state,
            )
            data = jnp.concat((data, sequence_output), axis=1)
            output = output.at[:, sequence].set(sequence_output[:, -1])
        return output, ("placeholder",)

    def _init_models(
        self,
    ) -> tuple[TransformerModel, train_state.TrainState, MemoryInterface]:
        key1, key2, self.model_config.prng_key = jax.random.split(
            self.model_config.prng_key, num=3
        )

        # init memory
        memory_model = self.model_config.memory_class()

        # init read and write heads
        read_heads = [
            self.model_config.read_head_class(*self.SINGLE_MEMORY_SHAPE)
            for _ in range(self.model_config.num_layers)
        ]
        write_heads = [
            self.model_config.write_head_class(*self.SINGLE_MEMORY_SHAPE)
            for _ in range(self.model_config.num_layers)
        ]

        # init backbone
        model = self.model_config.backbone_class(
            prng_key=key1,
            layers=self.model_config.num_layers,
            num_outputs=self.model_config.dim_model,
            read_heads=read_heads,
            write_heads=write_heads,
            dim_model=self.model_config.dim_model,
            num_heads=self.model_config.num_heads,
            dim_ff=self.model_config.dim_ff,
        )

        memory_weights = jnp.zeros(self.MEMORY_SHAPE)
        read_previous = jnp.ones(self.CONTROLLER_PREVIOUS_SHAPE)
        write_previous = jnp.ones(self.CONTROLLER_PREVIOUS_SHAPE)
        init_input = jnp.ones((2, self.model_config.dim_model))

        params = model.init(
            key2,
            init_input,
            memory_weights,
            read_previous,
            write_previous,
            memory_model,
        )  # Initialize parameters

        model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            tx=self.model_config.optimizer(self.model_config.learning_rate),
            params=params[globals.JAX.PARAMS],
        )

        return model, model_state, memory_model


if __name__ == "__main__":
    from Common.Checkpoint import CheckpointWrapper, TrainingMetadata
    from Common.globals import CURRICULUM, METADATA
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import NTMMemory
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.training_loop import train

    MEMORY_DEPTH = 8
    INPUT_SIZE = 8

    model_config = TransformerConfig(
        prng_key=jax.random.key(globals.JAX.RANDOM_SEED),
        learning_rate=1e-3,
        optimizer=optax.adamw,
        memory_class=NTMMemory,
        backbone_class=TransformerModel,
        read_head_class=NTMReadController,
        write_head_class=NTMWriteController,
        memory_M=MEMORY_DEPTH,
        memory_N=10,
        num_layers=4,
        dim_model=INPUT_SIZE,
        num_heads=4,
        dim_ff=250,
        max_sequence_len=10,
    )
    training_config = TransformerTrainingConfig(model_config)

    curriculum_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 2,
        CURRICULUM.CONFIGS.MAX: 10,
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
            f"{METADATA.COMPONENTS.CONTROLLERS.READ}{0}": training_config.model.read_heads[
                0
            ],
            f"{METADATA.COMPONENTS.CONTROLLERS.WRITE}{0}": training_config.model.write_heads[
                0
            ],
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
        "Transformer_copy_test", delete_existing=True
    )

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        training_metadata=training_metadata,
        num_epochs=1,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
        checkpoint_wrapper=checkpoint_wrapper,
    )

    from tqdm import tqdm

    from Visualization.MemoryWrappers import SequentialInferenceMemoryVisualizer

    training_config.memory_model = SequentialInferenceMemoryVisualizer(
        training_config.memory_model,
        save_dir="Transformer_copy_test",
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
    dataset_config = {globals.DATASETS.CONFIGS.CURRICULUM_SCHEDULER: curric}
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
            read_previous = (
                jnp.zeros((model_config.num_layers, TEST_MEMORY_WIDTH)).at[:, 0].set(1)
            )
            write_previous = (
                jnp.zeros((model_config.num_layers, TEST_MEMORY_WIDTH)).at[:, 0].set(1)
            )
            memory_weights = jnp.zeros(
                (model_config.num_layers, TEST_MEMORY_WIDTH, MEMORY_DEPTH)
            )

            # processing loop
            for sequence in range(1, data.shape[0] + 1):
                training_config.memory_model.update_step(
                    input_index=sequence - 1, output_index=None
                )

                output, memory_weights, read_previous, write_previous = (
                    training_config._prediction_fn(
                        data[:sequence],
                        memory_weights,
                        read_previous,
                        write_previous,
                        training_config.memory_model,
                        training_config.model_state.params,
                        training_config.model_state,
                    )
                )
            for sequence in range(data.shape[0] + 1):
                training_config.memory_model.update_step(
                    input_index=None, output_index=sequence
                )

                (
                    sequence_output,
                    memory_weights,
                    read_previous,
                    write_previous,
                ) = training_config._prediction_fn(
                    data,
                    memory_weights,
                    read_previous,
                    write_previous,
                    training_config.memory_model,
                    training_config.model_state.params,
                    training_config.model_state,
                )

                data = jnp.concat((data, sequence_output), axis=0)

                training_config.memory_model.add_output(
                    sequence_output, sequence, memory_weights[0]
                )

            training_config.memory_model.create_gif(loop=0, frame_duration=500)
