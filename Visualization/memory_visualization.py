from pathlib import Path

import jax.numpy as jnp
from jax import Array
from PIL import Image, ImageDraw, ImageFont

from Common.globals import VISUALIZATION

# color definitions (grayscale)
color_background = 255
color_foreground = 0


class SizingConstants:
    def __init__(self, pixel_scale: int = VISUALIZATION.DEFAULT_PIXEL_SCALE):
        self.pixel_scale = pixel_scale
        # padding on edges of image (left, right, top, bottom)
        self.padding_edges = [
            pixel_scale // 8,
            pixel_scale // 8,
            pixel_scale // 8,
            pixel_scale // 8,
        ]
        # padding between memory blocks
        self.padding_interior = pixel_scale // 8

        self.font_size = pixel_scale // 4
        self.draw_font = ImageFont.load_default(size=self.font_size)
        self.font_size_small = pixel_scale // 6
        self.draw_font_small = ImageFont.load_default(size=self.font_size_small)

        self.line_width = max(pixel_scale // 64, 1)

        self.arrow_padding = pixel_scale // 2
        self.label_padding = pixel_scale


def plot_memory_state_comparison(
    memory_1: Array,
    memory_2: Array,
    transpose: bool = False,
    save_location: str | None = None,
    annotation: list[str] = ["Target", "Output"],
    pixel_scale=VISUALIZATION.DEFAULT_PIXEL_SCALE,
) -> Image.Image:
    assert len(annotation) == 2, "Must have the exactly two annotations"
    memory_img_1 = plot_memory_state(
        memory_1, transpose=transpose, annotation=annotation[0], pixel_scale=pixel_scale
    )
    memory_img_2 = plot_memory_state(
        memory_2, transpose=transpose, annotation=annotation[1], pixel_scale=pixel_scale
    )

    if transpose:
        img = Image.new(
            "L", (memory_img_1.width + memory_img_2.width, memory_img_1.height)
        )
        img.paste(memory_img_1, (0, 0))
        img.paste(memory_img_2, (memory_img_1.width, 0))
    else:
        img = Image.new(
            "L", (memory_img_1.width, memory_img_1.height + memory_img_2.height)
        )
        img.paste(memory_img_1, (0, 0))
        img.paste(memory_img_2, (0, memory_img_1.height))

    if save_location:
        img.save(get_save_path(save_location, VISUALIZATION.IMG_EXTENSION))

    return img


def plot_memory_states_gif(
    memories: list[Array],
    attentions: list[Array] | None | list[None] = None,
    transpose: bool = False,
    save_location: str | None = None,
    loop: int | None = None,
    frame_duration: int = VISUALIZATION.DEFAULT_FRAME_DURATION,
    annotate_frame: bool = False,
    attention_label: str | None = VISUALIZATION.NAMES.ATTENTION,
    pixel_scale=VISUALIZATION.DEFAULT_PIXEL_SCALE,
) -> list[Image.Image]:
    if attentions is not None:
        assert len(memories) == len(
            attentions
        ), "Must have the same number of memory states and attentions"
    else:
        attentions = [None] * len(memories)

    memory_state_gif: list[Image.Image] = []
    for i in range(len(memories)):
        annotation = f"State {i}" if annotate_frame else None
        memory_state_gif.append(
            plot_memory_state(
                memories[i],
                attention=attentions[i],
                transpose=transpose,
                annotation=annotation,
                attention_label=attention_label,
                pixel_scale=pixel_scale,
            )
        )

    if save_location:
        save_location = get_save_path(save_location, VISUALIZATION.GIF_EXTENSION)
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
    attention_label: str | None = VISUALIZATION.NAMES.ATTENTION,
    small_font: bool = True,
    pixel_scale=VISUALIZATION.DEFAULT_PIXEL_SCALE,
) -> Image.Image:
    assert (
        len(attention.shape) == 1
    ), "Memory state visualization assumes a 1-dimensional attention"

    sc = SizingConstants(pixel_scale)
    font = sc.draw_font_small if small_font else sc.draw_font

    def color_from_value(value: float):
        # clip to (0,1)
        value = min(max(value, 0), 1)
        # negative interpolation
        min_value = 224
        max_value = 32
        value_color = int(min_value - (value * (min_value - max_value)))
        return value_color

    # make sure the attention is normalized
    # attention = jnp.divide(attention, jnp.sum(attention))

    # create new image with an extra row of blocks (and optionally extra X padding for labeling annotation vs memory)
    width_padding = 0
    if attention_label is not None:
        width_padding = sc.pixel_scale

    img2 = Image.new(
        "L",
        (img.width + width_padding, img.height + sc.pixel_scale + sc.padding_interior),
        color_background,
    )
    img2.paste(img, (width_padding, sc.pixel_scale + sc.padding_interior))

    draw = ImageDraw.ImageDraw(img2)

    # draw the attention blocks
    for i in range(attention.shape[0]):
        val = float(attention[i].item())
        x0 = (
            sc.padding_edges[0]
            + i * (sc.pixel_scale + sc.padding_interior)
            + width_padding
        )
        y0 = sc.padding_edges[2]
        draw.rectangle(
            [x0, y0, x0 + sc.pixel_scale, y0 + sc.pixel_scale],
            fill=color_from_value(val),
        )

    # draw a line separating attention from memory
    draw.line(
        [
            (0, sc.padding_edges[2] + sc.pixel_scale + sc.padding_interior // 2),
            (
                img2.width,
                sc.padding_edges[2] + sc.pixel_scale + sc.padding_interior // 2,
            ),
        ],
        fill=color_foreground,
        width=sc.line_width,
    )

    # notate which blocks are attention and which are memory
    if attention_label is not None:
        draw.text(
            (
                sc.padding_edges[0],
                sc.padding_edges[2] + sc.pixel_scale // 2 - sc.font_size_small // 2,
            ),
            attention_label,
            font=font,
            fill=color_foreground,
        )
        draw.text(
            (
                sc.padding_edges[0],
                sc.padding_edges[2]
                + sc.pixel_scale
                + sc.padding_interior
                + sc.pixel_scale // 2
                - sc.font_size_small // 2,
            ),
            VISUALIZATION.NAMES.MEMORY,
            font=font,
            fill=color_foreground,
        )

    return img2


def plot_memory_state(
    memory: Array,
    attention: Array | None = None,
    transpose: bool = False,
    save_location: str | None = None,
    annotation: str | None = None,
    attention_label: str | None = VISUALIZATION.NAMES.ATTENTION,
    small_attention_font: bool = True,
    pixel_scale=VISUALIZATION.DEFAULT_PIXEL_SCALE,
) -> Image.Image:
    sc = SizingConstants(pixel_scale)

    assert (
        len(memory.shape) == 2
    ), "Memory state visualization assumes a 2-dimensional memory"

    if transpose:
        memory = memory.transpose()

    def color_from_value(value: float):
        # clip to (0,1)
        value = min(max(value, 0), 1)
        # negative interpolation
        min_value = 224
        max_value = 32
        value_color = int(min_value - (value * (min_value - max_value)))
        return value_color

    N, M = memory.shape
    width = (
        sc.padding_edges[0]
        + sc.padding_edges[1]
        + N * sc.pixel_scale
        + (N - 1) * sc.padding_interior
    )
    height = (
        sc.padding_edges[2]
        + sc.padding_edges[3]
        + M * sc.pixel_scale
        + (M - 1) * sc.padding_interior
    )

    img = Image.new("L", (width, height))

    draw = ImageDraw.ImageDraw(img)

    # draw white background
    draw.rectangle([0, 0, width, height], fill=color_background)

    for i in range(N):
        for j in range(M):
            val = float(memory[i, j].item())
            x0 = sc.padding_edges[0] + i * (sc.pixel_scale + sc.padding_interior)
            y0 = sc.padding_edges[2] + j * (sc.pixel_scale + sc.padding_interior)
            draw.rectangle(
                [x0, y0, x0 + sc.pixel_scale, y0 + sc.pixel_scale],
                fill=color_from_value(val),
            )
    if attention is not None:
        img = plot_fuzzy_attention(
            img,
            attention,
            attention_label=attention_label,
            pixel_scale=sc.pixel_scale,
            small_font=small_attention_font,
        )

    if annotation is not None:
        # make a bigger image for the annotation
        img2 = Image.new(
            "L", (img.width, img.height + sc.font_size * 2), color_background
        )
        img2.paste(img, (0, sc.font_size * 2))
        img = img2

        draw = ImageDraw.ImageDraw(img)

        draw.text(
            (sc.padding_edges[0], sc.font_size // 2),
            annotation,
            font=sc.draw_font,
            fill=color_foreground,
        )

    if save_location:
        img.save(get_save_path(save_location, VISUALIZATION.IMG_EXTENSION))

    return img


def get_save_path(save_location: str, suffix: str):
    save_loc_path = Path(save_location)
    if save_loc_path.suffix != suffix:
        save_loc_path = save_loc_path.with_suffix(suffix)

    if not save_loc_path.is_absolute():
        save_loc_path = Path(VISUALIZATION.OUTPUT_DIR / save_loc_path)

    save_loc_path.parent.mkdir(parents=True, exist_ok=True)
    return str(save_loc_path)


if __name__ == "__main__":
    test_memory = jnp.array(
        [
            [1, 0.5, 0, 1, 0, 0, 0, 0],
            [0, 1, 0.5, 0, 1, 0, 0, 0],
            [0, 0, 1, 0.5, 0, 1, 0, 0],
            [0, 0, 0, 1, 0.5, 0, 1, 0],
        ]
    ).transpose()

    test_memory_comparison = jnp.array(
        [
            [0, 0, 0, 1, 0.5, 0, 1, 0],
            [0, 0, 1, 0.5, 0, 1, 0, 0],
            [0, 1, 0.5, 0, 1, 0, 0, 0],
            [1, 0.5, 0, 1, 0, 0, 0, 0],
        ]
    ).transpose()

    test_memory_attention = jnp.array([0, 0, 0, 0.1, 0.3, 0.1, 0.5, 0])

    test_memory_states = [
        jnp.array(
            [
                [1, 0.5, 0, 1, 0, 0, 0, 0],
                [0, 1, 0.5, 0, 1, 0, 0, 0],
                [0, 0, 1, 0.5, 0, 1, 0, 0],
                [0, 0, 0, 1, 0.5, 0, 1, 0],
            ]
        ).transpose(),
        jnp.array(
            [
                [0, 1, 0.5, 0, 1, 0, 0, 0],
                [0, 0, 1, 0.5, 0, 1, 0, 0],
                [0, 0, 0, 1, 0.5, 0, 1, 0],
                [0, 0, 0, 0, 1, 0.5, 0, 1],
            ]
        ).transpose(),
        jnp.array(
            [
                [0, 0, 1, 0.5, 0, 1, 0, 0],
                [0, 0, 0, 1, 0.5, 0, 1, 0],
                [0, 0, 0, 0, 1, 0.5, 0, 1],
                [1, 0, 0, 0, 0, 1, 0.5, 0],
            ]
        ).transpose(),
        jnp.array(
            [
                [0, 0, 0, 1, 0.5, 0, 1, 0],
                [0, 0, 0, 0, 1, 0.5, 0, 1],
                [1, 0, 0, 0, 0, 1, 0.5, 0],
                [0, 1, 0, 0, 0, 0, 1, 0.5],
            ]
        ).transpose(),
        jnp.array(
            [
                [0, 0, 0, 0, 1, 0.5, 0, 1],
                [1, 0, 0, 0, 0, 1, 0.5, 0],
                [0, 1, 0, 0, 0, 0, 1, 0.5],
                [0.5, 0, 1, 0, 0, 0, 0, 1],
            ]
        ).transpose(),
    ]

    memory_state_image = plot_memory_state(
        test_memory,
        attention=test_memory_attention,
        annotation="Memory visualization test",
        # attention_label=None,
        pixel_scale=128,
        save_location="test_mem.png",
    )
    # memory_state_image.show()

    memory_comparison_image = plot_memory_state_comparison(
        test_memory, test_memory_comparison, save_location="test_mem_comparison"
    )

    memory_states_gif = plot_memory_states_gif(
        test_memory_states, save_location="./test.gif", loop=0, annotate_frame=True
    )
