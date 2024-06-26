import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

from Common import globals
from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
from Memories.NTM_graves2014 import Memory


def init_train_state(
    model, random_key: jax.Array, shape: tuple[int, ...], learning_rate: float
) -> train_state.TrainState:
    variables = model.init(random_key, jnp.ones(shape), jnp.ones(shape))
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=variables[globals.JAX.PARAMS]
    )


def train_step(
    read_state: train_state.TrainState,
    write_state: train_state.TrainState,
    batch: jnp.ndarray,
    previous_state: jax.Array,
):
    def loss_fn(read_params, write_params):
        write_state.apply_fn({globals.JAX.PARAMS: write_params}, batch, previous_state)
        predictions = read_state.apply_fn(
            {globals.JAX.PARAMS: read_params}, batch, previous_state
        )
        loss = optax.losses.squared_error(predictions=predictions, targets=batch)
        return loss

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss), grads = gradient_fn(read_state.params, write_state.params)
    read_state = read_state.apply_gradients(grads=grads)
    write_state = write_state.apply_gradients(grads=grads)
    return read_state, write_state, loss


def train_and_eval(read_state, write_state, epochs, shape):
    previous_state = jnp.ones(shape)
    pbar = tqdm(range(1, epochs + 1))
    data_key = jax.random.key(globals.JAX.RANDOM_SEED)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}:")
        batch_key, data_key = jax.random.split(data_key)
        batch = jax.random.uniform(batch_key, shape)
        read_state, write_state, loss = train_step(
            read_state, write_state, batch, previous_state
        )
        pbar.postfix(loss=loss)


if __name__ == "__main__":
    test_n = 8
    test_m = 9
    test_model_feature_size = 10
    learning_rate = 1e-4
    batch_size = 32
    shape = (1, test_n)

    memory_model = Memory(test_n, test_m)
    read_controller = NTMReadController(memory_model)
    write_controller = NTMWriteController(memory_model)

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2 = jax.random.split(rng_key)

    read_controller_state = init_train_state(
        read_controller, key1, shape, learning_rate
    )
    write_controller_state = init_train_state(
        write_controller, key2, shape, learning_rate
    )

    print(f"{write_controller_state=}")

    train_and_eval(read_controller_state, write_controller_state, 2, shape)
