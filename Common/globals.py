"""
Global definitions of hard coded variables.
"""


# JAX related variables
class JAX:
    PARAMS = "params"
    RANDOM_SEED = 137


# Machine Specific Variables
class MACHINES:
    class GRAVES2014:
        NAME = "NTM graves 2014"
        MEMORY_BIAS = "memory_bias"
        LSTM_LAYER_STATE = "lstm_layer_state_"
        READ_CONTROLLER_BIAS = "read_controller_bias_"

    class YANG2014:
        NAME = "LANTM yang 2014"


# Visualization
class VISUALIZATION:
    OUTPUT_DIR = "Visualization_Outputs/"
    GIF_EXTENSION = ".gif"
    IMG_EXTENSION = ".png"


# Training Configuration (Hyperparameters)
class CONFIG:
    EPOCHS = "epochs"
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    OPTIMIZER = "optimizer"
    OPTIMIZER_ADAM = "adam"
    VAL_PERIOD = "validation_period_in_epochs"


# Metrics
class METRICS:
    LOSS = "loss"
    ACCURACY = "accuracy"


# WandB
class WANDB:
    class PROJECTS:
        INITIAL_SETUP = "neural-turing-machines"
        CODE_TESTING = "code-testing"

    class JOBS:
        TRAIN = "train"
        EVAL = "eval"

    class TAGS:
        LSTM = "lstm"
        TRANSFORMER = "transformer"
        RNN = "rnn"
        FNN = "fnn"
        LIE_ACCESS = "lie access"
        CODE_TESTING = "code testing"

    class LOGS:
        TRAIN = "train"
        VAL = "val"
        TEST = "test"
