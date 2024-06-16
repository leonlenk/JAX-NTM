from Common import common

import jax
from flax import linen as nn


# TODO may need to make the hidden state a learned bias vector?
class LSTMModel(nn.Module):
    """Basic stacked LSTM for controlling an NTM"""

    features: int
    layers: int
    seed: int = common.RANDOM_SEED

    def setup(self):
        lstm_layer = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast=common.JAX_PARAMS,
            split_rngs={common.JAX_PARAMS: False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm_layers = [lstm_layer(self.features) for _ in range(self.layers)]

    @nn.remat
    def __call__(self, x):
        for lstm_layer in self.lstm_layers:
            # TODO may need to not re-init carry/hidden each time?
            carry, hidden = lstm_layer.initialize_carry(
                jax.random.key(self.seed), x[:, 0].shape
            )
            (carry, hidden), x = lstm_layer((carry, hidden), x)

        return x


# basic test cases
if __name__ == "__main__":
    import jax.numpy as jnp

    layers = 4
    features = 32
    batch_size = 8
    input_features = 7
    input_length = 12

    x = jnp.ones((batch_size, input_length, input_features))
    model = LSTMModel(features=features,layers=layers)
    params = model.init(jax.random.key(common.RANDOM_SEED), x)
    y = model.apply(params, x)

    assert len(params[common.JAX_PARAMS]) == layers
    assert y.shape == (batch_size, input_length, features)
