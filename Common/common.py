"""
Global definitions of hard coded variables.
"""

# JAX Strings
JAX_PARAMS = "params"

# Random Seed
RANDOM_SEED = 137

# Models
MODEL_DNN = "dnn"

# Machine Specific Variables
GRAVES2014 = "NTM graves 2014"
GRAVES2014_MEMORY_BIAS = "memory_bias"
GRAVES2014_LSTM_LAYER_STATE = "lstm_layer_state_"
GRAVES2014_READ_CONTROLLER_BIAS = "read_controller_bias_"
YANG2014 = "LANTM yang 2014"

# Visualization
VISUALIZATION_OUTPUT_DIR = "Visualization_Outputs/"
VISUALIZATION_GIF_EXTENSION = ".gif"
VISUALIZATION_IMG_EXTENSION = ".png"

# Training Configuration (Hyperparameters)
CONFIG_EPOCHS = "epochs"
CONFIG_LEARNING_RATE = "learning_rate"
CONFIG_BATCH_SIZE = 16
CONFIG_OPTIMIZER = "optimizer"
CONFIG_OPTIMIZER_ADAM = "adam"
CONFIG_VAL_PERIOD = "validation_period_in_epochs"

# Metrics
METRIC_LOSS = "loss"
METRIC_ACCURACY = "accuracy"

# WandB
WANDB_PROJECT_NAME = "neural-turing-machines"
WANDB_JOB_TRAIN = "train"
WANDB_JOB_EVAL = "eval"
WANDB_TAG_LSTM = "lstm"
WANDB_TAG_TRANSFORMER = "transformer"
WANDB_TAG_RNN = "rnn"
WANDB_TAG_FNN = "fnn"
WANDB_TAG_LIE_ACCESS = "lie access"
WANDB_LOG_TRAIN = "train"
WANDB_LOG_VAL = "val"
WANDB_LOG_TEST = "test"
