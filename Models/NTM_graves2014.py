import jax
from flax import linen as nn
from jax import Array

from Common import globals


class LSTMModel(nn.Module):
    """Basic stacked LSTM for controlling an NTM"""

    features: int
    layers: int
    seed: int = globals.JAX.RANDOM_SEED

    @nn.compact
    def __call__(self, input: Array, states=None) -> tuple[Array, list[Array]]:
        lstm_layer = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast=globals.JAX.PARAMS,
            split_rngs={globals.JAX.PARAMS: False},
            in_axes=1,
            out_axes=1,
        )

        new_states = []
        for i in range(self.layers):
            lstm = lstm_layer(self.features)
            if states is None:
                state = self.param(
                    f"{globals.MACHINES.GRAVES2014.LSTM_LAYER_STATE}{i}",
                    lstm.initialize_carry,
                    input[:, 0].shape,
                )
            else:
                state = states[i]

            state, input = lstm(state, input)
            new_states.append(state)

        return input, new_states


# basic test cases
if __name__ == "__main__":
    import jax.numpy as jnp

    layers = 4
    features = 32
    batch_size = 8
    input_features = 7
    input_length = 12

    x = jnp.ones((batch_size, input_length, input_features))
    model = LSTMModel(features=features, layers=layers)
    params = model.init(jax.random.key(globals.JAX.RANDOM_SEED), x)
    y, states = model.apply(params, x)

    assert len(params[globals.JAX.PARAMS]) == layers
    assert y.shape == (batch_size, input_length, features)
