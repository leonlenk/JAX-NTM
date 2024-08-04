import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import orbax.checkpoint as ocp

from Common.BackboneInterface import BackboneInterface
from Common.ControllerInterface import ControllerInterface
from Common.globals import CHECKPOINTS, METADATA
from Common.MemoryInterface import MemoryInterface
from Common.TrainingInterfaces import (
    CurriculumSchedulerInterface,
    DataEncoderInterface,
    DataloaderInterface,
    TrainingConfigInterface,
)


@dataclass
class TrainingMetadata:
    Backbone: dict[str, BackboneInterface] | None = None
    Controller: dict[str, ControllerInterface] | None = None
    Memory: dict[str, MemoryInterface] | None = None
    DataEncoder: dict[str, DataEncoderInterface] | None = None
    CurriculumScheduler: dict[str, CurriculumSchedulerInterface] | None = None
    Dataloader: dict[str, DataloaderInterface] | None = None
    TrainingConfig: dict[str, TrainingConfigInterface] | None = None

    def get_metadata(self) -> dict:
        metadata = {}

        # TODO add other components
        if self.Backbone is not None:
            metadata[METADATA.COMPONENTS.BACKBONE] = {
                key: value.get_metadata() for key, value in self.Backbone.items()
            }
        if self.Controller is not None:
            metadata[METADATA.COMPONENTS.CONTROLLER] = {
                key: value.get_metadata() for key, value in self.Controller.items()
            }
        if self.Memory is not None:
            metadata[METADATA.COMPONENTS.MEMORY] = {
                key: value.get_metadata() for key, value in self.Memory.items()
            }
        if self.DataEncoder is not None:
            metadata[METADATA.COMPONENTS.DATA_ENCODER] = {
                key: value.get_metadata() for key, value in self.DataEncoder.items()
            }
        if self.CurriculumScheduler is not None:
            metadata[METADATA.COMPONENTS.CURRICULUM_SCHEDULER] = {
                key: value.get_metadata()
                for key, value in self.CurriculumScheduler.items()
            }
        if self.Dataloader is not None:
            metadata[METADATA.COMPONENTS.DATALOADER] = {
                key: value.get_metadata() for key, value in self.Dataloader.items()
            }
        if self.TrainingConfig is not None:
            metadata[METADATA.COMPONENTS.TRAINING_CONFIG] = {
                key: value.get_metadata() for key, value in self.TrainingConfig.items()
            }

        return metadata


class CheckpointWrapper:
    def __init__(
        self,
        save_path: str | None = None,
        options=None,
        delete_existing: bool = False,
    ):
        """Initializes the checkpoint system
        :param save_path: path to checkpoint save folder
            if save_path is not absolute, it will be appended to the /Checkpoints/ folder
        :param options: a pre-initialized parameter of type orbax.checkpoint.CheckpointManagerOptions
        """
        self.save_path: Path = Path(save_path) if save_path else Path("")
        if not self.save_path.is_absolute():
            # get the checkpoints directory /Checkpoints/
            common_directory = os.path.abspath(os.path.dirname(__file__) + "/..")
            checkpoints_dir = os.path.join(common_directory, CHECKPOINTS.DIR)
            self.save_path = Path(checkpoints_dir) / self.save_path

        if delete_existing:
            if self.save_path.is_dir():
                shutil.rmtree(str(self.save_path))

        if options is None:
            options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)

        self.mngr = ocp.CheckpointManager(
            self.save_path,
            options=options,
            item_names=(CHECKPOINTS.STATE, CHECKPOINTS.METADATA),
        )

        self.current_step = 0

    def save_checkpoint(
        self,
        state: Any,
        metadata: dict = {},
        step_increment: int = 1,
    ):
        args = {
            CHECKPOINTS.STATE: ocp.args.StandardSave(state),  # type: ignore
            CHECKPOINTS.METADATA: ocp.args.JsonSave(metadata),  # type: ignore
        }

        self.current_step += step_increment
        self.mngr.save(
            self.current_step,
            args=ocp.args.Composite(**args),
        )

    def load_checkpoint(
        self,
        step: int | None = None,
        abstract_pytree=None,
    ) -> Any:
        """Loads a checkpoint
        Access the state and metadata with
            checkpoint = load_checkpoint()
            state = checkpoint.state
            metadata = checkpoint.metadata

        :param step: which checkpoint step to load, defaults to the latest step
        :param abstract_pytree: an example state of the correct shape to load in
            may be necessary if the checkpoint manager has not yet saved a state of the same shape
        """
        if step is None:
            step = self.mngr.latest_step()

        assert step is not None, "No checkpoint found"

        self.current_step = step

        if abstract_pytree is not None:
            args = {
                CHECKPOINTS.STATE: ocp.args.StandardRestore(abstract_pytree),  # type: ignore
                CHECKPOINTS.METADATA: ocp.args.JsonRestore(),  # type: ignore
            }
            return self.mngr.restore(step, args=ocp.args.Composite(**args))

        return self.mngr.restore(step)

    def saved_steps(self) -> Sequence[int]:
        return self.mngr.all_steps()


if __name__ == "__main__":
    import optax

    from Backbones.NTM_graves2014 import LSTMModel
    from Common import globals
    from Common.globals import CURRICULUM
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import NTMMemory
    from Training.Backbones.NTM_graves2014 import LSTMConfig, LSTMTrainingConfig
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.training_loop import train

    MEMORY_DEPTH = 12

    model_config = LSTMConfig(
        learning_rate=1e-3,
        optimizer=optax.adam,
        memory_class=NTMMemory,
        backbone_class=LSTMModel,
        read_head_class=NTMReadController,
        write_head_class=NTMWriteController,
        memory_M=MEMORY_DEPTH,
        memory_N=8,
        num_layers=3,
        input_features=12,
    )
    training_config = LSTMTrainingConfig(model_config)

    curriculum_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 1,
        CURRICULUM.CONFIGS.MAX: 10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }
    curric = CurriculumSchedulerZaremba2014(curriculum_config)
    dataset_config = {globals.DATASETS.CONFIGS.CURRICULUM_SCHEDULER: curric}
    dataset = CopyLoader(
        batch_size=2,
        num_batches=1,
        memory_depth=MEMORY_DEPTH,
        options=dataset_config,
    )

    import jax

    from Common.globals import JAX
    from Common.TrainingInterfaces import DataEncoderStub
    from Memories import LANTM_yang2017

    learning_rate = 5e-3

    memory = LANTM_yang2017.LieAccessMemory(
        jax.random.key(JAX.RANDOM_SEED),
        (MEMORY_DEPTH,),
        optax.adam(learning_rate),
    )

    # initialize Rn addition with euclidean distance metric and softmax similarity
    memory.set_memory_functions(
        LANTM_yang2017.interpolation_Rn_function,
        LANTM_yang2017.group_action_Rn_addition,
        LANTM_yang2017.metric_euclidean_distance,
        LANTM_yang2017.similarity_softmax,
    )
    data_encoder = DataEncoderStub(12)
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
        # Memory={ str(0): memory, str(1): training_config.memory_model},
        Memory={str(0): training_config.memory_model},
        DataEncoder={str(0): data_encoder},
        CurriculumScheduler={str(0): curric},
        Dataloader={str(0): dataset},
        TrainingConfig={str(0): training_config},
    )
    # print_json(training_metadata.get_metadata())
    # exit()

    # print(f'{training_config.model_state}')

    checkpoint_wrapper = CheckpointWrapper("test_checkpoint", delete_existing=True)
    # checkpoint_wrapper.save_checkpoint(
    #     training_config.model_state, step=0, metadata=training_metadata.get_metadata()
    # )
    # restored_state = checkpoint_wrapper.load_checkpoint(
    #     step=0, abstract_pytree=training_config.model_state
    # )
    # print_json(restored_state.metadata)
    # training_config.model_state = restored_state.state

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        training_metadata=training_metadata,
        num_epochs=2,
        train_dataset=dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
        checkpoint_wrapper=checkpoint_wrapper,
        # use_wandb=True,
    )

    restored_state = checkpoint_wrapper.load_checkpoint(
        step=2, abstract_pytree=training_config.model_state
    )
    # print_json(restored_state.metadata)
