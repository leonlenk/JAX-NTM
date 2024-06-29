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
    memory_state: train_state.TrainState,
    memory_model: MemoryInterface,
    random_key: jax.Array,
    shape: tuple[int, ...],
    batch_shape,
    learning_rate: float,
) -> train_state.TrainState:
    variables = model.init(
        random_key, jnp.ones(batch_shape), jnp.ones(shape), memory_state, memory_model
    )
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=variables[globals.JAX.PARAMS]
    )


def train_step(
    read_state: train_state.TrainState,
    write_state: train_state.TrainState,
    memory_state: train_state.TrainState,
    memory_model: MemoryInterface,
    batch: jnp.ndarray,
    previous_state: jax.Array,
):
    def loss_fn(read_params, write_params, memory_params):
        write_state.apply_fn(
            {globals.JAX.PARAMS: write_params},
            batch,
            previous_state,
            memory_state,
            memory_model,
        )

        predictions = read_state.apply_fn(
            {globals.JAX.PARAMS: read_params},
            batch,
            previous_state,
            memory_state,
            memory_model,
        )[0]

        loss = jnp.mean(
            optax.losses.squared_error(predictions=predictions, targets=batch)
        )
        return loss

    gradient_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))
    loss, (read_grads, write_grads, memory_grads) = gradient_fn(
        read_state.params, write_state.params, memory_state.params
    )

    memory_state = memory_state.apply_gradients(grads=memory_grads)
    read_state = read_state.apply_gradients(grads=read_grads)
    write_state = write_state.apply_gradients(grads=write_grads)
    return read_state, write_state, memory_state, loss


def train_and_eval(
    read_state, write_state, memory_state, memory_model, epochs, shape, batch_shape
):
    previous_state = jnp.ones(shape)
    pbar = tqdm(range(1, epochs + 1))
    data_key = jax.random.key(globals.JAX.RANDOM_SEED)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}:")
        batch_key, data_key = jax.random.split(data_key)
        batch = jax.random.uniform(batch_key, batch_shape)
        read_state, write_state, memory_state, loss = train_step(
            read_state, write_state, memory_state, memory_model, batch, previous_state
        )
        pbar.set_postfix(loss=f"{loss:.2f}")


if __name__ == "__main__":
    test_n = 8
    test_m = 9
    test_model_feature_size = 10
    learning_rate = 1e-4
    batch_size = 32
    shape = (1, test_n)
    batch_shape = (1, test_m)
    num_epochs = 1000

    memory_model = Memory(test_n, test_m)
    read_controller = NTMReadController(*memory_model.size())
    write_controller = NTMWriteController(*memory_model.size())

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2, key3 = jax.random.split(rng_key, num=3)

    memory_variables = memory_model.init(key3, jnp.ones(shape), method=Memory.read)
    optimizer = optax.adam(learning_rate)
    memory_state = train_state.TrainState.create(
        apply_fn=memory_model.apply,
        tx=optimizer,
        params=memory_variables[globals.JAX.PARAMS],
    )

    read_controller_state = init_train_state(
        read_controller,
        memory_state,
        memory_model,
        key1,
        shape,
        batch_shape,
        learning_rate,
    )

    write_controller_state = init_train_state(
        write_controller,
        memory_state,
        memory_model,
        key2,
        shape,
        batch_shape,
        learning_rate,
    )

    train_and_eval(
        read_controller_state,
        write_controller_state,
        memory_state,
        memory_model,
        num_epochs,
        shape,
        batch_shape,
    )
