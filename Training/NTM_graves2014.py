from typing import Callable, Type

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from Common.BackboneInterface import BackboneInterface, PreviousState
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface
from Common.TrainingInterfaces import ModelConfigInterface, TrainingConfigInterface


class ModelConfig(ModelConfigInterface):
    def __init__(
        self,
        learning_rate: int,
        optimizer: Callable,
        memory_class: Type[MemoryInterface],
        backbone_class: Type[BackboneInterface],
        read_head_class: Type[ControllerInterface],
        write_head_class: Type[ControllerInterface],
        memory_M: int,
        memory_N: int,
        num_layers: int,
        num_outputs: int,
        input_length: int,
        input_features: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.memory_class = memory_class
        self.backbone_class = backbone_class
        self.read_head_class = read_head_class
        self.write_head_class = write_head_class
        self.memory_M = memory_M
        self.memory_N = memory_N
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.input_length = input_length
        self.input_features = input_features


class TrainingConfig(TrainingConfigInterface):
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.model, self.model_state, self.memory_model = self._init_models(
            model_config
        )

    def train_step(self) -> None:
        pass

    def _init_models(
        self, model_config: ModelConfig
    ) -> tuple[BackboneInterface, train_state.TrainState, MemoryInterface]:
        MEMORY_SHAPE = (
            model_config.memory_N,
            model_config.memory_M,
        )
        MEMORY_WIDTH = (1, model_config.memory_N)

        rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
        key1, key2, key3 = jax.random.split(rng_key, num=3)

        # init memory
        memory_model = model_config.memory_class(
            key1,
            (1, *MEMORY_SHAPE),
            model_config.optimizer(model_config.learning_rate),
        )

        # init read and write heads
        read_head = model_config.read_head_class(*MEMORY_SHAPE)
        write_head = model_config.write_head_class(*MEMORY_SHAPE)

        # init backbone
        model = model_config.backbone_class(
            key2,
            model_config.memory_M,
            model_config.num_layers,
            model_config.num_outputs,
            read_head,
            write_head,
        )
        init_input = jnp.ones(
            (
                model_config.input_length,
                model_config.input_features,
            )
        )
        init_previous_state = PreviousState(
            memory_model.weights, jnp.ones(MEMORY_WIDTH), jnp.ones(MEMORY_WIDTH)
        )
        params = model.init(key3, init_input, memory_model, init_previous_state)
        model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            tx=optax.adam(model_config.learning_rate),
            params=params[globals.JAX.PARAMS],
        )

        return model, model_state, memory_model
