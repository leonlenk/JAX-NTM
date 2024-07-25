import jax

from Backbones.NTM_graves2014 import LSTMModel
from Backbones.Transformer import TransformerModel
from Common import globals
from Common.TrainingInterfaces import DataloaderConfig
from GUI import config_options
from Training.Backbones.NTM_graves2014 import LSTMConfig, LSTMTrainingConfig
from Training.Backbones.Transformer import TransformerConfig, TransformerTrainingConfig


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
                input_features=1,
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
    if config_options.CURRICULUM[request["curriculum"]] != "none":
        curriculum_config = {
            globals.CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
            globals.CURRICULUM.CONFIGS.MIN: int(request["cirriculum_min_length"]),
            globals.CURRICULUM.CONFIGS.MAX: int(request["cirriculum_max_length"]),
            globals.CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
            globals.CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
            globals.CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
        }
        curriculum = config_options.CURRICULUM[request["curriculum"]](curriculum_config)

    dataset_config = DataloaderConfig(
        curriculum_scheduler=curriculum, accuracy_tolerance=0.1
    )

    return training_config, dataset_config
