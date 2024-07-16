from typing import Callable, Type

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from Common import globals
from Common.BackboneInterface import BackboneInterface
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface
from Common.TrainingInterfaces import ModelConfigInterface, TrainingConfigInterface


class ModelConfig(ModelConfigInterface):
    def __init__(
        self,
        learning_rate: float,
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
    model_config: ModelConfig

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.MEMORY_SHAPE = (
            model_config.memory_N,
            model_config.memory_M,
        )
        self.MEMORY_WIDTH = (model_config.memory_N,)
        self.model, self.model_state, self.memory_model = self._init_models()

    def loss_fn(self, model_params, data, target, criterion):
        # batch the function
        def prediction_fn(
            data, memory_weights, read_previous, write_previous, read_data, memory_model
        ):
            data = jnp.concat((data, read_data), axis=-1)
            (
                (output, read_data, memory_weights, read_previous, write_previous),
                variables,
            ) = self.model_state.apply_fn(
                {globals.JAX.PARAMS: model_params},
                data,
                memory_weights,
                read_previous,
                write_previous,
                memory_model,
                mutable=["state"],
            )
            return output, read_data, memory_weights, read_previous, write_previous

        batched_prediction_fn = jax.vmap(prediction_fn, in_axes=(0, 0, 0, 0, 0, None))

        # initial values
        read_previous = jnp.zeros((data.shape[0],) + self.MEMORY_WIDTH).at[:, 0].set(1)
        write_previous = jnp.zeros((data.shape[0],) + self.MEMORY_WIDTH).at[:, 0].set(1)
        read_data = jnp.zeros((data.shape[0], self.model_config.memory_M))
        memory_weights = jnp.zeros((data.shape[0],) + self.MEMORY_SHAPE)

        # processing loop
        for sequence in range(data.shape[1]):
            output, read_data, memory_weights, read_previous, write_previous = (
                batched_prediction_fn(
                    data[:, sequence],
                    memory_weights,
                    read_previous,
                    write_previous,
                    read_data,
                    self.memory_model,
                )
            )

        output = jnp.empty_like(target)
        for sequence in range(target.shape[1]):
            (
                sequence_output,
                read_data,
                memory_weights,
                read_previous,
                write_previous,
            ) = batched_prediction_fn(
                jnp.zeros_like(data[:, 0]),
                memory_weights,
                read_previous,
                write_previous,
                read_data,
                self.memory_model,
            )
            output = output.at[:, sequence].set(sequence_output)

        return criterion(output, target), ("placeholder",)

    def train_step(self, data, target, criterion):
        gradient_fn = jax.value_and_grad(self.loss_fn, argnums=(0), has_aux=True)
        ((loss, (metric,)), model_grads) = gradient_fn(
            self.model_state.params, data, target, criterion
        )
        self.model_state = self.model_state.apply_gradients(grads=model_grads)
        return {globals.METRICS.LOSS: loss}

    def val_step(self, data):
        return {}

    def _init_models(
        self,
    ) -> tuple[BackboneInterface, train_state.TrainState, MemoryInterface]:
        rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
        key1, key2, key3 = jax.random.split(rng_key, num=3)

        # init memory
        memory_model = self.model_config.memory_class()

        # init read and write heads
        read_head = self.model_config.read_head_class(*self.MEMORY_SHAPE)
        write_head = self.model_config.write_head_class(*self.MEMORY_SHAPE)

        # init backbone
        model = self.model_config.backbone_class(
            key2,
            self.model_config.memory_M,
            self.model_config.num_layers,
            self.model_config.num_outputs,
            read_head,
            write_head,
        )
        init_input = jnp.ones(
            (self.model_config.input_features + self.model_config.memory_M)
        )
        memory_weights = jnp.ones(self.MEMORY_SHAPE)
        params = model.init(
            key3,
            init_input,
            memory_weights,
            jnp.ones(self.MEMORY_WIDTH),
            jnp.ones(self.MEMORY_WIDTH),
            memory_model,
        )
        model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            tx=optax.adam(self.model_config.learning_rate),
            params=params[globals.JAX.PARAMS],
        )

        return model, model_state, memory_model
