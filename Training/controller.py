import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

from Common import globals
from Common.MemoryInterface import MemoryInterface
from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
from Memories.NTM_graves2014 import NTMMemory


def init_train_state(
    model,
    memory_model: MemoryInterface,
    random_key: jax.Array,
    shape: tuple[int, ...],
    batch_shape,
    learning_rate: float,
) -> train_state.TrainState:
    variables = model.init(
        random_key,
        jnp.ones(batch_shape),
        jnp.ones(shape),
        memory_model.weights,
        memory_model,
    )
    optimizer = optax.adam(learning_rate)
    # print(f'{variables[globals.JAX.PARAMS]=}')
    return train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=variables[globals.JAX.PARAMS]
    )


def train_step(
    read_state: train_state.TrainState,
    write_state: train_state.TrainState,
    memory_model: MemoryInterface,
    batch: jnp.ndarray,
    previous_state: jax.Array,
):
    def predict_fn(read_params, write_params, memory_weights, batch):
        written_memory_weights, write_memory_addresses = write_state.apply_fn(
            {globals.JAX.PARAMS: write_params},
            batch,
            previous_state,
            memory_weights,
            memory_model,
        )

        predictions, read_memory_addresses = read_state.apply_fn(
            {globals.JAX.PARAMS: read_params},
            batch,
            previous_state,
            written_memory_weights,
            memory_model,
        )

        return predictions, read_memory_addresses, write_memory_addresses

    batched_predict_fn = jax.vmap(predict_fn, in_axes=(None, None, None, 0))

    def loss_fn(read_params, write_params, memory_weights, batch):
        predictions, read_memory_addresses, write_memory_addresses = batched_predict_fn(
            read_params, write_params, memory_weights, batch
        )

        loss = jnp.mean(
            optax.losses.squared_error(predictions=predictions, targets=batch)
        )

        return loss, (read_memory_addresses, write_memory_addresses)

    gradient_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)
    (
        (loss, (read_addresses, write_addresses)),
        (read_grads, write_grads, memory_grads),
    ) = gradient_fn(read_state.params, write_state.params, memory_model.weights, batch)

    # print(read_addresses)
    # print(write_addresses)

    memory_model.apply_gradients(memory_grads)
    read_state = read_state.apply_gradients(grads=read_grads)
    write_state = write_state.apply_gradients(grads=write_grads)
    return read_state, write_state, memory_model, loss


def train_and_eval(read_state, write_state, memory_model, epochs, shape, batch_shape):
    previous_state = jnp.ones(shape)
    pbar = tqdm(range(1, epochs + 1))

    data_key = jax.random.key(globals.JAX.RANDOM_SEED)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}:")

        batch_key, data_key = jax.random.split(data_key)
        batch = jax.random.uniform(batch_key, batch_shape)

        read_state, write_state, memory_model, loss = train_step(
            read_state, write_state, memory_model, batch, previous_state
        )
        pbar.set_postfix(loss=f"{loss:.3f}")

    return read_state, write_state, memory_model


def test_batch(
    read_state: train_state.TrainState,
    write_state: train_state.TrainState,
    memory_model: MemoryInterface,
    batch: jnp.ndarray,
    previous_state: jax.Array,
):
    written_memory_weights, write_memory_addresses = write_state.apply_fn(
        {globals.JAX.PARAMS: write_state.params},
        batch,
        previous_state,
        memory_model.weights,
        memory_model,
    )

    predictions, read_memory_addresses = read_state.apply_fn(
        {globals.JAX.PARAMS: read_state.params},
        batch,
        previous_state,
        written_memory_weights,
        memory_model,
    )

    # print(f'{write_memory_addresses=}')
    # print(f'{read_memory_addresses=}')
    # print(f'{batch=}')
    # print(f'{predictions=}')

    return predictions


if __name__ == "__main__":
    from Common.MemoryInterface import MemoryVisualizerWrapper

    test_n = 8
    test_m = 12
    test_model_feature_size = 10
    learning_rate = 5e-3
    batch_size = 32
    shape = (1, test_n)
    single_input_shape = (1, test_m)
    batch_shape = (batch_size, 1, test_m)
    num_epochs = 300

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2, key3, key4 = jax.random.split(rng_key, num=4)

    memory_model = NTMMemory(key3, (1, test_n, test_m), optax.adam(learning_rate))
    read_controller = NTMReadController(test_n, test_m)
    write_controller = NTMWriteController(test_n, test_m)

    read_controller_state = init_train_state(
        read_controller,
        memory_model,
        key1,
        shape,
        single_input_shape,
        learning_rate,
    )

    write_controller_state = init_train_state(
        write_controller,
        memory_model,
        key2,
        shape,
        single_input_shape,
        learning_rate,
    )

    read_controller_state, write_controller_state, memory_model = train_and_eval(
        read_controller_state,
        write_controller_state,
        memory_model,
        num_epochs,
        shape,
        batch_shape,
    )

    # test out the memory visualization wrapper with a smaller number of epochs
    # outputs to Visualization_Outputs/training_test/

    memory_model = MemoryVisualizerWrapper(
        memory_model, save_dir="training_test", delete_existing=True
    )

    for i in range(2):
        input_key, rng_key = jax.random.split(rng_key)
        input = jax.random.uniform(input_key, single_input_shape)
        previous_state = jnp.ones(shape)
        preds = test_batch(
            read_controller_state,
            write_controller_state,
            memory_model,
            input,
            previous_state,
        )
