from typing import Callable

import jax.numpy as jnp
from jax import Array

from Common import globals
from Common.MemoryInterface import MemoryInterface


class MemoryLocation:
    """Individual units of memory for the LANTM
    Each memory location has a key, memory vector, and strength
    """

    def __init__(self, key: Array, memory_vector: Array, strength: float) -> None:
        self.key = key
        self.memory_vector = memory_vector
        self.strength = strength


# Helper functions
def read_from_weights(memory_state: list[MemoryLocation], weights: Array) -> Array:
    """Calculates the read vector
    :param memory_state: current memory state
    :param weights: weight for each memory state, must be the same length as memory_state

    :return: the weighted sum of each memory vector
    """
    read_vectors = []
    for memory_location, weight in zip(memory_state, weights):
        read_vectors.append(jnp.multiply(memory_location.memory_vector, weight))
    return jnp.mean(jnp.array(read_vectors), axis=0)


def normalize_arith_mean_similarity(
    memory_state: list[MemoryLocation],
    head_state: Array,
    similarity_function: Callable[[Array, Array, float], float],
) -> Array:
    """Calculates address weights
    :param memory_state: current memory state
    :param head_state: current head state vector
    :param similarity_function: function which takes in (head state, key, strength) and returns the weight

    :return: a list of normalized weights for each memory location

    s_i * S(q, k_i)
    /
    sum_j(s_j * S(q, k_j))
    """
    weights_list: list[float] = []
    for memory_location in memory_state:
        weights_list.append(
            similarity_function(
                head_state, memory_location.key, memory_location.strength
            )
        )

    weights: Array = jnp.array(weights_list)
    return jnp.divide(weights, jnp.sum(weights))


# Similarity functions
def similarity_softmax(
    memory_state: list[MemoryLocation],
    head_state: Array,
    metric: Callable[[Array, Array], float],
    temperature: float,
) -> Array:
    """Implements softmax similarity
    :param memory_state: current memory state
    :param head_state: current head state vector
    :param metric: function which takes in (head state, key) and returns a measure of distance
    :param temperature: temperature of softmax, lower = sharper addressing

    :return: the weight array

    s_i * e^(-d(q,k_i)^2 / T)
    /
    sum_j(s_j * e^(-d(q,k_j)^2 / T))
    """

    def softmax(head_state: Array, key: Array, strength: float) -> float:
        distance = metric(head_state, key)
        return strength * jnp.exp(-(distance**2) / temperature).item()

    weights: Array = normalize_arith_mean_similarity(memory_state, head_state, softmax)
    return weights


def similarity_inverse_square(
    memory_state: list[MemoryLocation],
    head_state: Array,
    metric: Callable[[Array, Array], float],
) -> Array:
    """Implements inverse square similarity
    :param memory_state: current memory state
    :param head_state: current head state vector
    :param metric: function which takes in (head state, key) and returns a measure of distance

    :return: the weight array

    s_i / (d(q,k_i)^2 + ε)
    /
    sum_j(s_j / d(q,k_j)^2 + ε))
    """

    def inverse_square(head_state: Array, key: Array, strength: float) -> float:
        distance = metric(head_state, key)
        return strength / (distance**2 + globals.JAX.EPSILON)

    weights: Array = normalize_arith_mean_similarity(
        memory_state, head_state, inverse_square
    )
    return weights


# Metric functions
def metric_euclidean_distance(head_state: Array, key: Array) -> float:
    """Implements euclidean distance metric
    :param head_state: current head state vector
    :param key: key vector

    :return: measure of distance

    sum_i((x_i - y_i)^2)^(1/2)
    """
    return jnp.linalg.norm(jnp.subtract(head_state, key))


# Interpolation Functions
def interpolation_Rn_function(
    head_state: Array, interpolation_vector: Array, interpolation_gate: float
) -> Array:
    """Implements natural Rn interpolation
    :param head_state: previous head state vector
    :param interpolation_vector: vector "r" updating the state
    :param interpolation_gate: scalar "g" in [0,1] indicating how strongly to interpolation
        0 means the interpolation will be ignored
        1 means the previous head state will be ignored

    q * (1 - g) + r * g
    """
    return jnp.add(
        jnp.multiply(head_state, 1 - interpolation_gate),
        jnp.multiply(interpolation_vector, interpolation_gate),
    )


# Group action functions
def group_action_Rn_addition(head_state: Array, action: Array) -> Array:
    """Implements Rn addition group
    :param head_state: previous head state vector
    :param action: shift to add to previous head state

    :return: new head_state

    q = (x, y, ...)
    a = (α, β, ...)
    a * q = (x + α, y + β, ...)
    """
    return jnp.add(head_state, action)


# TODO group action, metric, interpolation functions for SO3 on S2


class Memory(MemoryInterface):
    """Memory interface for NTM from Yang 2017 (arXiv:1611.02854).

    The memory operates as follows
        1. Reading
            1.1 Receive the following inputs
                1.1.1 previous head state "q_p" (the address of the previous read)
                1.1.2 interpolation vector "r"
                1.1.3 interpolation gate "g" (scalar)
                1.1.4 action "a" (the Lie group operation to update q)
            1.2 Calculate the new head state "q" from q_p, r, g, and a
            1.3 Calculate the weights "w" between q and each memory location's key "k" using a defined similarity function "S"
            1.4 Return a weighted sum of memory vectors "v" (dot product of v and w)

        # TODO is this really the write method? Doesn't this effectively prevent writing new values to old locations?
            Should we try pre-populating the memory state and having write attention weighted like read is?
        2. Writing
            2.1 Receive the following inputs
                2.1.1 previous head state "q_p" (the address of the previous write)
                2.1.2 interpolation vector "r"
                2.1.3 interpolation gate "g" (scalar)
                2.1.4 action "a" (the Lie group operation to update q)
                2.1.5 memory vector to write "v_w"
                2.1.6 strength of new memory "s_w"
            2.2 Calculate the new head state "q" from q_p, r, g, and a
            2.3 Add a new memory location to the memory state with (key, memory vector, strength) of (q, v_w, s_w)


    The memory state is a dictionary "Σ" of memory locations
        Each memory location has a key "k", memory vector "v", and strength "s"
            k has a length of D
            v has a length of M
            s is a scalar


    The memory is defined by:
        The memory depth "M" (length of memory vectors)

        An interpolation function "I" which
            Updates a head state "q" based on an interpolation vector "r" and interpolation gate "g"
            This is for moving to an absolute address, rather than relative
            r indicates the address, g indicates the magnitude of the move

            Example:
                R2 linear interpolation
                    q = (x, y)
                    r = (a, b)
                    I(q, r, g) = q * (1 - g) + r * g


        A Lie group "L" which
            Operates in a space of dimension "D" (length of key vectors)
            Is restricted to a manifold of dimension "d" (length of head state)
            Defines the location-based addressing through an action "a" which
                Is defined in a space of dimension "A" (length of addressing operation)
                Operates "*" on the current head state "q"


            Examples:
                R2 addition
                    D = 2, d = 2, A = 2     # R2
                    q = (x, y)
                    a = (α, β)
                    L: a * q = (x + α, y + β)

                SO3 rotation
                    D = 3, d = 2, A = 3     # Unit sphere in R3
                    q = (φ, θ)              # Unit vector in R3
                    a = (α, β, γ)           # Unit vector axis of rotation (α, β), angle to rotate γ
                    L:

        A similarity function "S" which
            For a given head state "q" and memory state "Σ"
            S calculates each memory location's weight
                based on the similarity of q to each key "k", as well as each key's strength "s"
            S is defined by
                A metric "d" on the Lie group
                The weighting function "W"

            Example:
                d(a, b) = ||a - b||^2                   # euclidean distance in Rn
                W_i(q, Σ, T) =                          # softmax weighted by s for a given temperature "T"
                    s_i * e^(-d(q,k_i)^2 / T)
                    /
                    sum_j(s_i * e^(-d(q,k_j)^2 / T))
    """

    def __init__(
        self,
        rng_key,
        memory_shape,
        optimizer,
    ):
        self.rng_key = rng_key

        self.memory_shape = memory_shape

        self.memory_state: list[MemoryLocation] = []

    def set_memory_functions(
        self,
        interpolation_fn: Callable[[Array, Array, float], Array],
        group_action_fn: Callable[[Array, Array], Array],
        metric_fn: Callable[[Array, Array], float],
        similarity_fn: Callable[..., Array],
    ):
        """Initializes functions defining the memory accessing scheme
        :param interpolation_fn: maps (previous head state, interpolation vector, interpolation gate) to (new head state)
            called before the group action
        :param group_action_fn: maps (previous head state, action) to (new head state)
        :param metric_fn: maps (head_state, key) to (distance)
        :param similarity_fn: maps (memory state, head state, metric, additional parameters) to (read vector)
        """
        self.group_action_fn = group_action_fn
        self.metric_fn = metric_fn
        self.interpolation_fn = interpolation_fn
        self.similarity_fn = similarity_fn

    def apply_gradients(self, gradients):
        pass

    # TODO update memoryinterface for more generic read/write/address functions

    def read(self, weights: Array) -> Array:
        return read_from_weights(self.memory_state, weights)

    def write(self, head_state: Array, memory_vector: Array, strength: float):
        # TODO do something different if head state (the new key) exactly matches an existing key?
        self.memory_state.append(MemoryLocation(head_state, memory_vector, strength))

    def address(
        self,
        previous_head_state: Array,
        interpolation_vector: Array,
        interpolation_gate: float,
        action: Array,
        *args,
    ) -> tuple[Array, Array]:
        head_state = self.interpolation_fn(
            previous_head_state, interpolation_vector, interpolation_gate
        )
        head_state = self.group_action_fn(head_state, action)
        return head_state, self.similarity_fn(
            self.memory_state, head_state, self.metric_fn, *args
        )


if __name__ == "__main__":
    import jax
    import optax

    memory_depth = 8
    learning_rate = 5e-3

    memory = Memory(
        jax.random.key(globals.JAX.RANDOM_SEED),
        (memory_depth,),
        optax.adam(learning_rate),  # TODO nothing is done with optimizer right now...
    )

    # initialize Rn addition with euclidean distance metric and softmax similarity
    memory.set_memory_functions(
        interpolation_Rn_function,
        group_action_Rn_addition,
        metric_euclidean_distance,
        similarity_softmax,
    )

    temperature = 1
    head_state = jnp.zeros(memory_depth)

    action = jnp.ones(memory_depth)
    memory_vector = jnp.arange(memory_depth).astype(jnp.float32)
    strength = 0.7
    interpolation_vector = jnp.ones(memory_depth)
    interpolation_scalar = 0.2
    head_state, weights = memory.address(
        head_state, interpolation_vector, interpolation_scalar, action, temperature
    )
    # print(f'{head_state=}')
    memory.write(head_state, memory_vector, strength)

    action = jnp.zeros((memory_depth,))
    action = action.at[0].set(1)
    memory_vector = jnp.ones(memory_depth)
    strength = 0.9
    head_state, weights = memory.address(
        head_state, interpolation_vector, interpolation_scalar, action, temperature
    )
    # print(f'{head_state=}')
    memory.write(head_state, memory_vector, strength)

    head_state, weights = memory.address(
        head_state, interpolation_vector, interpolation_scalar, action, temperature
    )
    read_vector = memory.read(weights)
    # print(f'{read_vector=}')

    # for x in memory.memory_state:
    #     print(f'{x.key=}\n{x.memory_vector=}\n{x.strength=}\n')

    assert read_vector.shape[0] == memory_depth
