import traceback

"""
Global definitions of hard coded variables.
"""


# JAX related variables
class JAX:
    PARAMS = "params"

    RANDOM_SEED = 137
    EPSILON = 1e-6


class CHECKPOINTS:
    DIR = "Checkpoints"
    STATE = "state"
    METADATA = "metadata"


class METADATA:
    NAME = "name"
    SEED = "seed"
    CONFIG = "config"
    MEMORY_DEPTH = "memory_depth"
    MEMORY_LENGTH = "memory_length"
    SPLIT = "split"
    SINGLE_CURR_PER_BATCH = "single_curriculum_level_per_batch"
    ACCURACY_TOLERANCE = "accuracy_tolerance"

    # TODO add other components
    class COMPONENTS:
        BACKBONE = "backbone"
        CONTROLLER = "controller"

        class CONTROLLERS:
            READ = "read"
            WRITE = "write"

        MEMORY = "memory"
        DATA_ENCODER = "data_encoder"
        CURRICULUM_SCHEDULER = "curriculum_scheduler"
        DATALOADER = "dataloader"

        class DATALOADERS:
            TRAIN = "train"
            VAL = "val"
            TEST = "test"

        TRAINING_CONFIG = "training_config"


# Machine Specific Variables
class MACHINES:
    class GRAVES2014:
        NAME = "NTM graves 2014"
        MEMORY_BIAS = "memory_bias"
        LSTM_LAYER_STATE = "lstm_layer_state_"
        READ_CONTROLLER_BIAS = "read_controller_bias_"

        class MEMORY:
            N = "num_memory_locations"
            M = "memory_depth"

    class YANG2014:
        NAME = "LANTM yang 2014"
        GROUP_ACTION = "group_action"
        METRIC = "metric"
        INTERPOLATION = "interpolation"
        SIMILARITY = "similarity"


# Visualization
class VISUALIZATION:
    OUTPUT_DIR = "Visualization_Outputs/"
    GIF_EXTENSION = ".gif"
    IMG_EXTENSION = ".png"
    DEFAULT_PIXEL_SCALE = 128
    DEFAULT_FRAME_DURATION = 500

    class NAMES:
        DEFAULT = "memory"
        READ = "Read"
        WRITE = "Write"
        ADDRESS = "Address"
        INFERENCE_GIF = "inference_gif"
        ATTENTION = "Attention"
        MEMORY = "Memory"

    class SEQUENTIAL_INFERENCE:
        INPUT = "Input"
        OUTPUT = "Output"
        TARGET = "Target"


# Training Configuration (Hyperparameters)
class CONFIG:
    EPOCHS = "epochs"
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    NUM_BATCHES = "number_of_batches"
    OPTIMIZER = "optimizer"
    OPTIMIZER_ADAM = "adam"
    VAL_PERIOD = "validation_period_in_epochs"


class MODELS:
    MEMORY = "memory_model"
    READ_CONTROLLER = "read_controller"
    WRITE_CONTROLLER = "write_controller"
    MACHINE = "machine"
    BASE = "base_model"
    OPTIMIZER = "optimizer"

    class LSTM:
        FEATURES = "features"
        LAYERS = "layers"
        NUM_OUTPUTS = "num_outputs"


# Datasets
class DATASETS:
    CACHE_LOCATION = "Datasets/cache"
    CACHE_EXTENSION = ".npz"

    DEFAULT_ACCURACY_TOLERANCE = 1e-2

    class ADD:
        class CONFIGS:
            INTERLEAVED = "interleaved"

    class SPLITS:
        TRAIN = "train"
        VAL = "val"
        TEST = "test"

    class ENCODERS:
        VOCABULARY = "vocabulary"

        class CONFIGS:
            CACHE_DIR = "cache_dir"

    class BABI:
        NAME = "bAbI"
        DATA_PATH = "tasks_1-20_v1-2"
        CACHE = "cache"
        DATA_EXTENSION = ".txt"
        PUNCTUATION_MARKS = [".", "?", ","]

        class CONFIGS:
            SET = "set"

        class SETS:
            EN = "en"
            EN_10K = "en-10k"
            EN_VALID = "en-valid"
            EN_VALID_10K = "en-valid-10k"
            HN = "hn"
            HN_10K = "hn-10k"
            SHUFFLED = "shuffled"
            SHUFFLED_10K = "shuffled-10k"

        class TASKS:
            TASKS = [x for x in range(1, 21)]
            ID = {x: f"qa{x}" for x in range(1, 21)}
            NAME = {
                1: "single-supporting-fact",
                2: "two-supporting-facts",
                3: "three-supporting-facts",
                4: "two-arg-relations",
                5: "three-arg-relations",
                6: "yes-no-questions",
                7: "counting",
                8: "lists-sets",
                9: "simple-negation",
                10: "indefinite-knowledge",
                11: "basic-coreference",
                12: "conjunction",
                13: "compound-coreference",
                14: "time-reasoning",
                15: "basic-deduction",
                16: "basic-induction",
                17: "positional-reasoning",
                18: "size-reasoning",
                19: "path-finding",
                20: "agents-motivations",
            }


# Curriculum
class CURRICULUM:
    # initialization options
    class CONFIGS:
        MIN = "min difficulty"
        MAX = "max difficulty"
        LOSS_THRESHOLD = "loss threshold"
        ACCURACY_THRESHOLD = "accuracy threshold"
        RANDOM_SEED = "seed"

        class ZAREMBA2014:
            """From Zaremba and Sutskever 2014 arXiv:1410.4615
            The curriculum level "D" starts at the minimum level "min".
            It increments by 1 towards the maximum level "max"
            every time a chosen metric passes a threshold on the current difficulty.

            Given a curriculum level "D", the difficulty "d" of the next example is selected as follows.

            "p1" percent of the time:
                "d" is selected uniformly from ["min", "max"]
            "p2" percent of the time:
                "e" is selected from a geometric distribution with success probability 1/2, and
                "d" is selected uniformly from ["min", "D" + "e"]
            "p3" percent of the time:
                "e" is selected from a geometric distribution with success probability 1/2, and
                "d" is set to "D" + "e"

            The values given in the paper are
            {
                "p1": 0.10,
                "p2": 0.25,
                "p3": 0.65,
            }
            """

            P1 = "p1"
            P2 = "p2"
            P3 = "p3"


# Metrics
class METRICS:
    LOSS = "loss"
    ACCURACY = "accuracy"
    EPOCH = "epoch"
    CURRICULUM_LEVEL = "curriculum_level"


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


# TODO clean this up...
def print_items(**kwargs):
    def func(var):
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]

    for item in kwargs.items():
        print(f"{item[0]} = {item[1]}")


def print_json(data):
    import json

    print(json.dumps(data, indent=4))
