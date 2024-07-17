import os
from pathlib import Path
from typing import Any, Sequence

import orbax.checkpoint as ocp

from Common.globals import JAX


class CheckpointWrapper:
    def __init__(
        self,
        save_path: str | None = None,
        options=None,
    ):
        self.save_path: Path = Path(save_path) if save_path else Path("")
        if not self.save_path.is_absolute():
            # get the checkpoints directory /Checkpoints/
            common_directory = os.path.abspath(os.path.dirname(__file__) + "/..")
            checkpoints_dir = os.path.join(common_directory, JAX.CHECKPOINTS.DIR)
            self.save_path = Path(checkpoints_dir) / self.save_path

        if options is None:
            options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)

        self.mngr = ocp.CheckpointManager(
            self.save_path, options=options, item_names=(JAX.STATE, JAX.METADATA)
        )

    def save_checkpoint(
        self,
        state: Any,
        step: int = 0,
        metadata=None,
    ):
        args = {
            JAX.STATE: ocp.args.StandardSave(state),  # type: ignore
        }
        if metadata is not None:
            args[JAX.METADATA] = ocp.args.JsonSave(metadata)  # type: ignore

        self.mngr.save(
            step,
            args=ocp.args.Composite(**args),
        )

    def load_checkpoint(
        self,
        step: int | None = None,
        abstract_pytree=None,
        metadata: bool = False,
    ) -> Any:
        if step is None:
            step = self.mngr.latest_step()

        assert step is not None, "No checkpoint found"
        if abstract_pytree is not None:
            args = {
                JAX.STATE: ocp.args.StandardRestore(abstract_pytree),  # type: ignore
            }
            if metadata:
                args[JAX.METADATA] = ocp.args.JsonRestore()  # type: ignore
            return self.mngr.restore(step, args=ocp.args.Composite(**args))
        return self.mngr.restore(step)

    def saved_steps(self) -> Sequence[int]:
        return self.mngr.all_steps()


if __name__ == "__main__":
    import optax

    from Backbone.NTM_graves2014 import LSTMModel
    from Common import globals
    from Common.globals import CURRICULUM
    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Datasets.copy import CopyLoader
    from Memories.NTM_graves2014 import Memory
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014
    from Training.NTM_graves2014 import ModelConfig, TrainingConfig
    from Training.training_loop import train

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

    # print(f'{training_config.model_state}')

    checkpointer = CheckpointWrapper("test_checkpoint")
    checkpointer.save_checkpoint(
        training_config.model_state, step=1, metadata={"test": 1}
    )
    restored_state = checkpointer.load_checkpoint(
        step=1, abstract_pytree=training_config.model_state, metadata=True
    )
    # print(f'{new_state}')
    training_config.model_state = restored_state.state

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
        config=dataset_config,
    )

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        num_epochs=1,
        train_dataset=dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
    )
