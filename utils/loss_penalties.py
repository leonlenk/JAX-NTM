import jax
import jax.numpy as jnp


def calculate_entropy(array: jax.Array, bin_width: float = 0.1):
    # Flatten the array
    flat_array = jnp.ravel(array)

    # Group values by rounding to nearest bin_width
    binned_array = jnp.round(flat_array / bin_width) * bin_width

    # Find unique binned values and their counts
    unique_elements, counts = jnp.unique(binned_array, return_counts=True)

    # Calculate probabilities
    total_elements = jnp.sum(counts)
    probabilities = counts / total_elements

    # Calculate entropy
    entropy_value = jnp.sum(jax.scipy.special.entr(probabilities))

    return entropy_value


def entropy_penalty(array: jax.Array, max_penalty=5, epsilon=1e-8):
    penalty = 1.0 / (calculate_entropy(array).item() + epsilon)
    clipped_penalty = min(penalty, max_penalty)
    return clipped_penalty


if __name__ == "__main__":
    assert calculate_entropy(jnp.array([0, 0])) == 0
    assert calculate_entropy(jnp.array([1, 0])) != 0
    print(entropy_penalty(jnp.array([0, 0])))
    print(entropy_penalty(jnp.array([1, 0])))
    print(entropy_penalty(jnp.array([1, 0, 0, 0, 0, 0, 0])))
    print(entropy_penalty(jnp.array([1, 0, 0, 0, 0, 0, 0.1])))
