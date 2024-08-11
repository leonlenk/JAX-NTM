from dataclasses import dataclass
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
    learning_rate: float
    optimizer: Callable
    memory_class: Type[MemoryInterface]
    backbone_class: Type[TransformerModel]
    read_head_class: Type[ControllerInterface]
    write_head_class: Type[ControllerInterface]
    memory_M: int
    memory_N: int
    num_layers: int
    num_heads: int
    dim_ff: int
    max_sequence_len: int
    random_seed: int
    hidden_dim: int

    def __post_init__(self):
        self.prng_key = jax.random.key(self.random_seed)


class TransformerTrainingConfig(TrainingConfigInterface):
    model_config: TransformerConfig

    def __init__(self, model_config: TransformerConfig) -> None:
        self.model_config = model_config
        self.model, self.model_state, self.memory_model = self._init_models()
        self._batched_run_model = jax.vmap(
            self.run_model, in_axes=(None, 0, None, None)
        )

    def run_model(self, model_params, data, output_shape, memory_width):
        # initial values
        read_previous = (
            jnp.zeros((model_config.num_layers, memory_width)).at[:, 0].set(1)
        )
        write_previous = (
            jnp.zeros((model_config.num_layers, memory_width)).at[:, 0].set(1)
        )
        memory_weights = jnp.zeros(
            (model_config.num_layers, memory_width, model_config.memory_M)
        )

        # processing loop
        for sequence in range(1, data.shape[0] + 1):
            self.memory_model.update_step(input_index=sequence - 1, output_index=None)
            (
                output,
                memory_weights,
                read_previous,
                write_previous,
            ) = self.model_state.apply_fn(
                {globals.JAX.PARAMS: model_params},
                data[:sequence],
                memory_weights,
                read_previous,
                write_previous,
                self.memory_model,
            )

        # prediciton loop
        output = jnp.empty(output_shape)
        prediction_input = jnp.zeros((1,) + output_shape[1:])
        for sequence in range(output_shape[0]):
            self.memory_model.update_step(input_index=None, output_index=sequence)
            (
                sequence_output,
                memory_weights,
                read_previous,
                write_previous,
            ) = self.model_state.apply_fn(
                {globals.JAX.PARAMS: model_params},
                prediction_input,
                memory_weights,
                read_previous,
                write_previous,
                self.memory_model,
            )
            prediction_input = sequence_output
            output = output.at[sequence].set(sequence_output[-1])
            self.memory_model.add_output(
                output_vector=sequence_output[-1],
                index=sequence,
                memory_weights=memory_weights[0],
            )
        return output

    def _init_models(
        self,
    ) -> tuple[TransformerModel, train_state.TrainState, MemoryInterface]:
        temp_n = 2
        MEMORY_SHAPE = (
            model_config.num_layers,
            temp_n,
            model_config.memory_M,
        )
        SINGLE_MEMORY_SHAPE = (
            temp_n,
            model_config.memory_M,
        )

        key1, self.model_config.prng_key = jax.random.split(
            self.model_config.prng_key, num=2
        )

        # init memory
        memory_model = self.model_config.memory_class()

        # init read and write heads
        read_heads = [
            self.model_config.read_head_class(*SINGLE_MEMORY_SHAPE)
            for _ in range(self.model_config.num_layers)
        ]
        write_heads = [
            self.model_config.write_head_class(*SINGLE_MEMORY_SHAPE)
            for _ in range(self.model_config.num_layers)
        ]

        # init backbone
        model = self.model_config.backbone_class(
            layers=self.model_config.num_layers,
            num_outputs=self.model_config.hidden_dim,
            read_heads=read_heads,
            write_heads=write_heads,
            dim_model=self.model_config.hidden_dim,
            num_heads=self.model_config.num_heads,
            dim_ff=self.model_config.dim_ff,
        )

        memory_weights = jnp.zeros(MEMORY_SHAPE)
        read_previous = jnp.ones((model_config.num_layers, temp_n))
        write_previous = jnp.ones((model_config.num_layers, temp_n))
        init_input = jnp.ones((2, self.model_config.hidden_dim))

        params = model.init(
            key1,
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
    from Common.TrainingInterfaces import DataloaderConfig
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import NTMMemory
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.training_loop import train

    MEMORY_DEPTH = 9
    INPUT_SIZE = 8

    model_config = TransformerConfig(
        learning_rate=1e-3,
        optimizer=optax.adamw,
        memory_class=NTMMemory,
        backbone_class=TransformerModel,
        read_head_class=NTMReadController,
        write_head_class=NTMWriteController,
        memory_M=MEMORY_DEPTH,
        memory_N=10,
        num_layers=2,
        hidden_dim=INPUT_SIZE,
        num_heads=1,
        dim_ff=250,
        max_sequence_len=20,
        random_seed=globals.JAX.RANDOM_SEED,
    )
    training_config = TransformerTrainingConfig(model_config)

    curriculum_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 2,
        CURRICULUM.CONFIGS.MAX: 4,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }
    curric = CurriculumSchedulerZaremba2014(curriculum_config)
    dataset_config = DataloaderConfig(curriculum_scheduler=curric)
    train_dataset = CopyLoader(
        batch_size=20,
        num_batches=1,
        memory_depth=INPUT_SIZE,
        config=dataset_config,
    )

    val_dataset = CopyLoader(
        batch_size=20,
        num_batches=1,
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

            training_config.run_model(
                training_config.model_state.params,
                data,
                target.shape,
                TEST_MEMORY_WIDTH,
            )

            training_config.memory_model.create_gif(loop=0, frame_duration=500)
