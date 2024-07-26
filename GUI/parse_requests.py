import jax

from Backbones.NTM_graves2014 import LSTMModel
from Backbones.Transformer import TransformerModel
from Common import globals
from Common.Checkpoint import TrainingMetadata
from Common.TrainingInterfaces import DataloaderConfig
from GUI import config_options
from Training.Backbones.NTM_graves2014 import LSTMConfig, LSTMTrainingConfig
from Training.Backbones.Transformer import TransformerConfig, TransformerTrainingConfig
from Training.training_loop import train


def process_training_request(request: dict):
    match request["model"]:
        case "lstm":
            model_config = LSTMConfig(
                learning_rate=float(request["learning_rate"]),
                optimizer=config_options.OPTIMIZERS[request["optimizer"]],
                memory_M=int(request["memory_m"]),
                memory_N=int(request["memory_n"]),
                memory_class=config_options.MEMORY_MODELS[request["memory_model"]],
                backbone_class=LSTMModel,
                read_head_class=config_options.READ_CONTROLLERS[
                    request["read_controller"]
                ],
                write_head_class=config_options.WRITE_CONTROLLERS[
                    request["write_controller"]
                ],
                num_layers=int(request["layers"]),
                input_features=int(request["data_input_size"]),
            )
            training_config = LSTMTrainingConfig(model_config)

        case "transfomer":
            model_config = TransformerConfig(
                learning_rate=float(request["learning_rate"]),
                optimizer=config_options.OPTIMIZERS[request["optimizer"]],
                memory_M=int(request["memory_m"]),
                memory_N=int(request["memory_n"]),
                prng_key=jax.random.key(request["random_seed"]),
                memory_class=config_options.MEMORY_MODELS[request["memory_model"]],
                backbone_class=TransformerModel,
                read_head_class=config_options.READ_CONTROLLERS[
                    request["read_controller"]
                ],
                write_head_class=config_options.WRITE_CONTROLLERS[
                    request["write_controller"]
                ],
                num_layers=int(request["layers"]),
                dim_model=int(request["transformer_dim_model"]),
                num_heads=int(request["transfomer_num_heads"]),
                dim_ff=int(request["transfomer_dim_ff"]),
                max_sequence_len=int(request["transfomer_max_sequence_length"]),
            )
            training_config = TransformerTrainingConfig(model_config)

    curriculum = None
    curriculumMetaData = None
    if config_options.CURRICULUM[request["curriculum"]] is not None:
        curriculum_config = {
            globals.CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: float(
                request["cirriculum_accuracy_threshold"]
            ),
            globals.CURRICULUM.CONFIGS.MIN: int(request["cirriculum_min_length"]),
            globals.CURRICULUM.CONFIGS.MAX: int(request["cirriculum_max_length"]),
            globals.CURRICULUM.CONFIGS.ZAREMBA2014.P1: float(request["zaremba_p1"]),
            globals.CURRICULUM.CONFIGS.ZAREMBA2014.P2: float(request["zaremba_p2"]),
            globals.CURRICULUM.CONFIGS.ZAREMBA2014.P3: float(request["zaremba_p3"]),
        }
        curriculum = config_options.CURRICULUM[request["curriculum"]](curriculum_config)
        curriculumMetaData = {str(0): curriculum}

    dataset_config = DataloaderConfig(
        curriculum_scheduler=curriculum,
        accuracy_tolerance=float(request["accuracy_tolerance"]),
    )

    train_dataset = config_options.DATA_SET[request["dataset"]](
        batch_size=int(request["train_batch_size"]),
        num_batches=int(request["train_num_batches"]),
        memory_depth=int(request["data_input_size"]),
        config=dataset_config,
    )

    val_dataset = config_options.DATA_SET[request["dataset"]](
        batch_size=int(request["val_batch_size"]),
        num_batches=int(request["val_num_batches"]),
        memory_depth=int(request["data_input_size"]),
        config=dataset_config,
    )

    training_metadata = TrainingMetadata(
        Backbone={str(0): training_config.model},
        Controller={
            f"{globals.METADATA.COMPONENTS.CONTROLLERS.READ}{i}": head
            for i, head in enumerate(training_config.model.read_heads)
        }
        | {
            f"{globals.METADATA.COMPONENTS.CONTROLLERS.WRITE}{i}": head
            for i, head in enumerate(training_config.model.write_heads)
        },
        Memory={str(0): training_config.memory_model},
        DataEncoder={},
        CurriculumScheduler=curriculumMetaData,
        Dataloader={
            globals.METADATA.COMPONENTS.DATALOADERS.TRAIN: train_dataset,
            globals.METADATA.COMPONENTS.DATALOADERS.VAL: val_dataset,
        },
        TrainingConfig={str(0): training_config},
    )

    train(
        project_name=globals.WANDB.PROJECTS.CODE_TESTING,
        training_config=training_config,
        training_metadata=training_metadata,
        num_epochs=int(request["num_epochs"]),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        wandb_tags=[globals.WANDB.TAGS.CODE_TESTING],
        use_wandb=(request["use_wandb"] == "yes"),
    )
