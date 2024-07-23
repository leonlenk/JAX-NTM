import optax

import Controllers
import Controllers.NTM_graves2014
import Memories.NTM_graves2014

OPTIMIZERS = {"adam": optax.adam, "sgd": optax.sgd, "adamw": optax.adamw}

MEMORY_MODELS = {
    "NTM": Memories.NTM_graves2014.NTMMemory,
}

READ_CONTROLLERS = {"NTM": Controllers.NTM_graves2014.NTMReadController}

WRITE_CONTROLLERS = {"NTM": Controllers.NTM_graves2014.NTMWriteController}
