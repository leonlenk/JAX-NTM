import jax
from flax import linen as nn

from Common.common import *


class LSTMModel(nn.Module):
    """Basic stacked LSTM for controlling the NTM"""
    features: int
    layers: int

    def setup(self):
        lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                               variable_broadcast="params",
                               split_rngs={"params": False},
                               in_axes=1, out_axes=1,
                               reverse=False)
        self.lstm_layers = [lstm_layer(self.features) for _ in range(self.layers)]
        
    @nn.remat    
    def __call__(self, x):
        for lstm_layer in self.lstm_layers:
            carry, hidden = lstm_layer.initialize_carry(jax.random.PRNGKey(RANDOM_SEED),x[:, 0].shape)
            (carry, hidden), x = lstm_layer((carry, hidden), x)

        return x
    

