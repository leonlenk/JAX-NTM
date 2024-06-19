from Common import common

import jax
from flax import linen as nn


class LSTMModel(nn.Module):
    """Basic stacked LSTM for controlling an NTM"""

    features: int
    layers: int
    seed: int = common.RANDOM_SEED

    @nn.compact
    def __call__(self, input, states=None):
        lstm_layer = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast=common.JAX_PARAMS,
            split_rngs={common.JAX_PARAMS: False},
            in_axes=1,
            out_axes=1,
        )

        new_states = []
        for i in range(self.layers):
            lstm = lstm_layer(self.features)
            if states is None:
                state = self.param(
                    f"{common.GRAVES2014_LSTM_LAYER_STATE}{i}",
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
    params = model.init(jax.random.key(common.RANDOM_SEED), x)
    y, states = model.apply(params, x)

    assert len(params[common.JAX_PARAMS]) == layers
    assert y.shape == (batch_size, input_length, features)
