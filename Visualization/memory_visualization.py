from pathlib import Path

import jax.numpy as jnp
from jax import Array
from PIL import Image, ImageDraw

from Common import common

# memory block display size
pixel_scale = 128
# padding on edges of image (left, right, top, bottom)
padding_edges = [pixel_scale // 8, pixel_scale // 8, pixel_scale // 2, pixel_scale // 8]
# padding between memory blocks
padding_interior = pixel_scale // 8


def plot_memory_states_gif(
    memories: list[Array],
    attentions: list[Array] | None | list[None] = None,
    transpose: bool = True,
    save_location: str | None = None,
    loop: int | None = None,
    frame_duration: int = 500,
) -> list[Image.Image]:
    if attentions is not None:
        assert len(memories) == len(
            attentions
        ), "Must have the same number of memory states and attentions"
    else:
        attentions = [None] * len(memories)

    memory_state_gif: list[Image.Image] = []
    for i in range(len(memories)):
        memory_state_gif.append(
            plot_memory_state(memories[i], attentions[i], transpose)
        )

    if save_location:
        save_location = get_save_path(save_location, common.VISUALIZATION_GIF_EXTENSION)
        memory_state_gif[0].save(
            save_location,
            save_all=True,
            append_images=memory_state_gif[1:],
            optimize=True,
            duration=frame_duration,
            loop=loop,
        )

    return memory_state_gif


def plot_fuzzy_attention(
    img: Image.Image,
    attention: Array,
    attention_color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    assert (
        len(attention.shape) == 1
    ), "Memory state visualization assumes a 1-dimensional attention"

    def color_from_value(value: float):
        min_value = 0
        max_value = 128
        value_color = int(min_value + value * (max_value - min_value))
        return attention_color + (value_color,)

    # make sure the attention is normalized
    attention = jnp.divide(attention, jnp.sum(attention))

    draw = ImageDraw.ImageDraw(img, "RGBA")

    for i in range(attention.shape[0]):
        val = float(attention[i])
        x0 = padding_edges[0] + i * (pixel_scale + padding_interior)
        y0 = padding_edges[2]
        draw.rectangle(
            [x0, y0, x0 + pixel_scale, img.size[1] - padding_edges[3]],
            fill=color_from_value(val),
        )

    return img


def plot_memory_state(
    memory: Array,
    attention: Array | None = None,
    transpose: bool = True,
    save_location: str | None = None,
) -> Image.Image:
    assert (
        len(memory.shape) == 2
    ), "Memory state visualization assumes a 2-dimensional memory"

    if transpose:
        memory = memory.transpose()

    # color definitions (grayscale)
    color_background = (255, 255, 255)

    def color_from_value(value: float):
        # clip to (0,1)
        value = min(max(value, 0), 1)
        # negative interpolation
        min_value = 224
        max_value = 32
        value_color = int(min_value - (value * (min_value - max_value)))
        return (value_color, value_color, value_color)

    N, M = memory.shape
    width = (
        padding_edges[0]
        + padding_edges[1]
        + N * pixel_scale
        + (N - 1) * padding_interior
    )
    height = (
        padding_edges[2]
        + padding_edges[3]
        + M * pixel_scale
        + (M - 1) * padding_interior
    )

    img = Image.new("RGB", (width, height))

    draw = ImageDraw.ImageDraw(img)

    # draw white background
    draw.rectangle([0, 0, width, height], fill=color_background)

    for i in range(N):
        for j in range(M):
            val = float(memory[i, j])
            x0 = padding_edges[0] + i * (pixel_scale + padding_interior)
            y0 = padding_edges[2] + j * (pixel_scale + padding_interior)
            draw.rectangle(
                [x0, y0, x0 + pixel_scale, y0 + pixel_scale], fill=color_from_value(val)
            )

    if attention is not None:
        img = plot_fuzzy_attention(img, attention)

    if save_location:
        img.save(get_save_path(save_location, common.VISUALIZATION_IMG_EXTENSION))

    return img


def get_save_path(save_location: str, suffix: str):
    save_loc_path = Path(save_location)
    if save_loc_path.suffix != suffix:
        save_loc_path = save_loc_path.with_suffix(suffix)

    if not save_loc_path.is_absolute():
        save_loc_path = Path(common.VISUALIZATION_OUTPUT_DIR / save_loc_path)

    save_loc_path.parent.mkdir(parents=True, exist_ok=True)
    return str(save_loc_path)


# TODO: add test cases
if __name__ == "__main__":
    test_memory = jnp.array(
        [
            [1, 0.5, 0, 1, 0, 0, 0, 0],
            [0, 1, 0.5, 0, 1, 0, 0, 0],
            [0, 0, 1, 0.5, 0, 1, 0, 0],
            [0, 0, 0, 1, 0.5, 0, 1, 0],
        ]
    )
    test_memory_attention = jnp.array([0, 0, 0, 0.1, 0.3, 0.1, 0.5, 0])

    test_memory_states = [
        jnp.array(
            [
                [1, 0.5, 0, 1, 0, 0, 0, 0],
                [0, 1, 0.5, 0, 1, 0, 0, 0],
                [0, 0, 1, 0.5, 0, 1, 0, 0],
                [0, 0, 0, 1, 0.5, 0, 1, 0],
            ]
        ),
        jnp.array(
            [
                [0, 1, 0.5, 0, 1, 0, 0, 0],
                [0, 0, 1, 0.5, 0, 1, 0, 0],
                [0, 0, 0, 1, 0.5, 0, 1, 0],
                [0, 0, 0, 0, 1, 0.5, 0, 1],
            ]
        ),
        jnp.array(
            [
                [0, 0, 1, 0.5, 0, 1, 0, 0],
                [0, 0, 0, 1, 0.5, 0, 1, 0],
                [0, 0, 0, 0, 1, 0.5, 0, 1],
                [1, 0, 0, 0, 0, 1, 0.5, 0],
            ]
        ),
        jnp.array(
            [
                [0, 0, 0, 1, 0.5, 0, 1, 0],
                [0, 0, 0, 0, 1, 0.5, 0, 1],
                [1, 0, 0, 0, 0, 1, 0.5, 0],
                [0, 1, 0, 0, 0, 0, 1, 0.5],
            ]
        ),
        jnp.array(
            [
                [0, 0, 0, 0, 1, 0.5, 0, 1],
                [1, 0, 0, 0, 0, 1, 0.5, 0],
                [0, 1, 0, 0, 0, 0, 1, 0.5],
                [0.5, 0, 1, 0, 0, 0, 0, 1],
            ]
        ),
    ]

    memory_state_image = plot_memory_state(test_memory, attention=test_memory_attention)
    memory_state_image.show()

    memory_states_gif = plot_memory_states_gif(
        test_memory_states, save_location="./test.gif", loop=0
    )
