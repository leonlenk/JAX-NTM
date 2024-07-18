from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from Common import globals
from Common.BackboneInterface import BackboneInterface
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface


class TransformerLayer(nn.Module):
    dim_model: int
    num_heads: int
    dim_ff: int
    read_head: ControllerInterface
    write_head: ControllerInterface

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        memory_weights: jax.Array,
        read_previous: jax.Array,
        write_previous: jax.Array,
        memory_model: MemoryInterface,
    ):
        # attention
        attention_output = nn.SelfAttention(
            num_heads=self.num_heads, qkv_features=self.dim_model
        )(x[None, ...])[0]
        x = nn.LayerNorm()(x + attention_output)

        # read data
        read_data, read_locations = self.read_head(
            x[-1],
            read_previous,
            memory_weights,
            memory_model,
        )

        # reshape and concat read_data with the input
        read_data = jnp.expand_dims(read_data, axis=0)
        read_data = jnp.tile(read_data, (x.shape[-2], 1))
        dense_input = jnp.concat((x, read_data), axis=-1)

        # dense layers
        ff_output = nn.relu(nn.Dense(self.dim_ff)(dense_input))
        ff_output = nn.Dense(self.dim_model)(ff_output)

        # write data
        memory_weights, write_locations = self.write_head(
            ff_output[-1],
            write_previous,
            memory_weights,
            memory_model,
        )

        # residual
        x = nn.LayerNorm()(x + ff_output)

        return x, memory_weights, read_locations, write_locations


class PositionalEncoding(nn.Module):
    dim_model: int
    max_sequence_len: int = 5000

    def setup(self):
        positional_encoding = np.zeros((self.max_sequence_len, self.dim_model))
        position = np.arange(0, self.max_sequence_len).reshape(-1, 1)
        normalization = np.exp(
            np.arange(0, self.dim_model, 2) * -(np.log(10000.0) / self.dim_model)
        )

        positional_encoding[:, 0::2] = np.sin(position * normalization)
        positional_encoding[:, 1::2] = np.cos(position * normalization)

        self.positional_encoding = jnp.array(positional_encoding)

    def __call__(self, x):
        return x + self.positional_encoding[: x.shape[0], :]


class TransformerModel(BackboneInterface):
    prng_key: jax.Array
    layers: int
    num_outputs: int
    read_heads: Sequence[ControllerInterface]
    write_heads: Sequence[ControllerInterface]

    dim_model: int
    num_heads: int
    dim_ff: int
    max_sequence_len: int = 5000

    def setup(self):
        self.positional_encoding = PositionalEncoding(
            self.dim_model, self.max_sequence_len
        )
        self.transformer_layers = [
            TransformerLayer(
                self.dim_model,
                self.num_heads,
                self.dim_ff,
                self.read_heads[i],
                self.write_heads[i],
            )
            for i in range(self.layers)
        ]
        self.final_layer = nn.Dense(self.num_outputs)

    def __call__(
        self, input, memory_weights, read_previous, write_previous, memory_model
    ):
        input = self.positional_encoding(input)

        # pass the input through the transformer layers
        for i in range(self.layers):
            input, new_memory_weights, new_read_previous, new_write_previous = (
                self.transformer_layers[i](
                    input,
                    memory_weights[i],
                    read_previous[i],
                    write_previous[i],
                    memory_model,
                )
            )
            memory_weights.at[i].set(new_memory_weights)
            read_previous.at[i].set(new_read_previous)
            write_previous.at[i].set(new_write_previous)

        input = self.final_layer(input)

        return input, memory_weights, read_previous, write_previous


if __name__ == "__main__":
    import optax
    from flax.training import train_state
    from jax.tree_util import tree_flatten

    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Memories.NTM_graves2014 import NTMMemory

    memory_n = 256
    memory_m = 12
    num_layers = 5
    dim_model = 512
    lr = 1e-3
    num_recursions = 2

    prng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2 = jax.random.split(prng_key, num=2)

    read_heads = [NTMReadController(memory_n, memory_m) for _ in range(num_layers)]
    write_heads = [NTMWriteController(memory_n, memory_m) for _ in range(num_layers)]
    memory_model = NTMMemory()
    model = TransformerModel(
        prng_key=key1,
        layers=num_layers,
        num_outputs=dim_model,
        read_heads=read_heads,
        write_heads=write_heads,
        dim_model=dim_model,
        num_heads=8,
        dim_ff=2048,
    )

    memory_weights = jnp.zeros((num_layers, memory_n, memory_m))
    read_previous = jnp.ones((num_layers, memory_n))
    write_previous = jnp.ones(((num_layers, memory_n)))

    init_input = jnp.ones((2, dim_model))

    params = model.init(
        key2, init_input, memory_weights, read_previous, write_previous, memory_model
    )  # Initialize parameters
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optax.adam(lr),
        params=params[globals.JAX.PARAMS],
    )

    def loss_fn(model_params):
        test_input = jnp.ones((2, dim_model))
        read_previous = jnp.ones((num_layers, memory_n))
        write_previous = jnp.ones(((num_layers, memory_n)))
        memory_weights = jnp.zeros((num_layers, memory_n, memory_m))

        # self, input, memory_weights, read_previous, write_previous, memory_model
        for _ in range(num_recursions):
            (
                test_input,
                memory_weights,
                read_previous,
                write_previous,
            ) = model_state.apply_fn(
                {globals.JAX.PARAMS: model_params},
                test_input,
                memory_weights,
                read_previous,
                write_previous,
                memory_model,
            )

        return jnp.sum(test_input)

    gradient_fn = jax.grad(loss_fn, argnums=(0))
    model_grads = gradient_fn(model_state.params)

    flat_grads, _ = tree_flatten(model_grads)
    for grad in flat_grads:
        assert jnp.nonzero(grad), "some of the gradients were zero!"

    print("passed all tests")
