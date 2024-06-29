import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

from Common import globals
from Common.MemoryInterface import MemoryInterface
from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
from Memories.NTM_graves2014 import Memory


def init_train_state(
    model,
    memory_weights: jax.Array,
    memory_model: MemoryInterface,
    random_key: jax.Array,
    shape: tuple[int, ...],
    batch_shape,
    learning_rate: float,
) -> train_state.TrainState:
    variables = model.init(
        random_key, jnp.ones(batch_shape), jnp.ones(shape), memory_weights, memory_model
    )
    optimizer = optax.adam(learning_rate)
    # print(f'{variables[globals.JAX.PARAMS]=}')
    return train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=variables[globals.JAX.PARAMS]
    )


def train_step(
    read_state: train_state.TrainState,
    write_state: train_state.TrainState,
    memory_weights: jax.Array,
    memory_model: MemoryInterface,
    batch: jnp.ndarray,
    previous_state: jax.Array,
):
    def loss_fn(read_params, write_params, memory_weights):
        memory_weights, memory_addresses = write_state.apply_fn(
            {globals.JAX.PARAMS: write_params},
            batch,
            previous_state,
            memory_weights,
            memory_model,
        )

        predictions, memory_addresses = read_state.apply_fn(
            {globals.JAX.PARAMS: read_params},
            batch,
            previous_state,
            memory_weights,
            memory_model,
        )

        loss = jnp.mean(
            optax.losses.squared_error(predictions=predictions, targets=batch)
        )
        return loss

    gradient_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))
    loss, (read_grads, write_grads, memory_grads) = gradient_fn(
        read_state.params, write_state.params, memory_weights
    )

    # TODO apply gradients to memory to update bias?
    # memory_state = memory_state.apply_gradients(grads=memory_grads)
    read_state = read_state.apply_gradients(grads=read_grads)
    write_state = write_state.apply_gradients(grads=write_grads)
    return read_state, write_state, memory_weights, loss


def train_and_eval(
    read_state, write_state, memory_weights, memory_model, epochs, shape, batch_shape
):
    previous_state = jnp.ones(shape)
    pbar = tqdm(range(1, epochs + 1))
    # TODO update training to get random.uniform batches to converge?
    # data_key = jax.random.key(globals.JAX.RANDOM_SEED)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}:")

        # TODO update training to get random.uniform batches to converge?
        # batch_key, data_key = jax.random.split(data_key)
        # batch = jax.random.uniform(batch_key, batch_shape)
        batch = jnp.ones(batch_shape)
        read_state, write_state, memory_weights, loss = train_step(
            read_state, write_state, memory_weights, memory_model, batch, previous_state
        )
        pbar.set_postfix(loss=f"{loss:.2f}")


if __name__ == "__main__":
    test_n = 8
    test_m = 9
    test_model_feature_size = 10
    learning_rate = 5e-3
    batch_size = 32
    shape = (1, test_n)
    batch_shape = (1, test_m)
    num_epochs = 1000

    memory_model = Memory()
    read_controller = NTMReadController(test_n, test_m)
    write_controller = NTMWriteController(test_n, test_m)

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2, key3 = jax.random.split(rng_key, num=3)

    memory_weights = jnp.zeros((1, test_n, test_m))

    memory_variables = memory_model.init(
        key3, memory_weights, jnp.ones(shape), method=Memory.read
    )
    optimizer = optax.adam(learning_rate)

    read_controller_state = init_train_state(
        read_controller,
        memory_weights,
        memory_model,
        key1,
        shape,
        batch_shape,
        learning_rate,
    )

    write_controller_state = init_train_state(
        write_controller,
        memory_weights,
        memory_model,
        key2,
        shape,
        batch_shape,
        learning_rate,
    )

    train_and_eval(
        read_controller_state,
        write_controller_state,
        memory_weights,
        memory_model,
        num_epochs,
        shape,
        batch_shape,
    )
