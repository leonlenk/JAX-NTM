import jax.numpy as jnp
from jax import Array
from PIL import Image, ImageDraw


def plot_memory_state(memory: Array, pixel_scale: int = 32) -> Image.Image:
    assert (
        len(memory.shape) == 2
    ), "Memory state visualization assumes a 2-dimensional memory"

    # color definitions (grayscale)
    color_background = 255

    def color_from_value(value: float):
        # clip to (0,1)
        value = min(max(value, 0), 1)
        # negative interpolation
        min_value = 224
        max_value = 32
        return int(min_value - (value * (min_value - max_value)))

    # padding on edge of image
    padding_edge = pixel_scale // 8
    # padding between memory blocks
    padding_interior = pixel_scale // 8

    N, M = memory.shape
    height = padding_edge * 2 + N * pixel_scale + (N - 1) * padding_interior
    width = padding_edge * 2 + M * pixel_scale + (M - 1) * padding_interior

    im = Image.new("L", (width, height))

    draw = ImageDraw.ImageDraw(im)

    # draw white background
    draw.rectangle([0, 0, width, height], fill=color_background)

    for i in range(N):
        for j in range(M):
            val = float(memory[i, j])
            y0 = padding_edge + i * (pixel_scale + padding_interior)
            x0 = padding_edge + j * (pixel_scale + padding_interior)
            draw.rectangle(
                [x0, y0, x0 + pixel_scale, y0 + pixel_scale], fill=color_from_value(val)
            )
    return im


# TODO: add test cases
if __name__ == "__main__":
    test_memory = jnp.array(
        [
            [1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
        ]
    )
    memory_state_image = plot_memory_state(test_memory)
    memory_state_image.show()
