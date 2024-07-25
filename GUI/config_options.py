import optax

import Controllers
import Controllers.NTM_graves2014
import Datasets.addition
import Datasets.babi
import Datasets.copy
import Memories.NTM_graves2014
import Training.Curriculum_zaremba2014

OPTIMIZERS = {"adam": optax.adam, "sgd": optax.sgd, "adamw": optax.adamw}

MEMORY_MODELS = {
    "NTM": Memories.NTM_graves2014.NTMMemory,
}

READ_CONTROLLERS = {"NTM": Controllers.NTM_graves2014.NTMReadController}

WRITE_CONTROLLERS = {"NTM": Controllers.NTM_graves2014.NTMWriteController}

DATA_SET = {
    "copy": Datasets.copy.CopyLoader,
    "babi": Datasets.babi.BabiLoader,
    "addition": Datasets.addition.BinaryAdditionLoader,
}

CURRICULUM = {
    "zaremba": Training.Curriculum_zaremba2014,
}
