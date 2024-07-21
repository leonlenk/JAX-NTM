import os
import shutil
from pathlib import Path

from jax import Array
from jax import numpy as jnp
from PIL import Image, ImageDraw

from Common.globals import VISUALIZATION
from Common.MemoryInterface import MemoryInterface
from Visualization import memory_visualization

# TODO make wrapper to plot read/write address vs time (iteration #)


class SequentialInferenceMemoryVisualizer(MemoryInterface):
    """Wrapper around memory to visualize sequential inference.

    At each step, will create an image containing the:
        input sequence (and which item, if any, is currently being inputted)
        output sequence (and which item, if any, is currently being outputted)
        target
        memory state
        read or write attention

    And at the end of inference, these images will be combined into a gif.


    Usage:
    memory = Memory()
    memory = SequentialInferenceMemoryVisualizer(memory, ...)

    # run this before each inference to set the inference-specific variables
    memory.set_up_inference(input, target, memory_length, memory_depth)
    """

    def __init__(
        self,
        wrapped_memory: MemoryInterface,
        save_dir: str | None = None,
        save_name: str | None = None,
        delete_existing: bool = False,
        pixel_scale=VISUALIZATION.DEFAULT_PIXEL_SCALE,
    ):
        self.wrapped_memory = wrapped_memory
        self.sc = memory_visualization.SizingConstants(pixel_scale)
        self.sc_small = memory_visualization.SizingConstants(pixel_scale // 2)

        self.save_dir: Path = Path(save_dir) if save_dir else Path("")

        if not self.save_dir.is_absolute():
            self.save_dir = Path(VISUALIZATION.OUTPUT_DIR) / self.save_dir
            self.save_dir = self.save_dir.resolve()

        self.save_name: str = save_name if save_name else VISUALIZATION.NAMES.DEFAULT

        if delete_existing:
            if self.save_dir.is_dir():
                shutil.rmtree(str(self.save_dir))

        if not os.path.exists(str(self.save_dir)):
            os.makedirs(str(self.save_dir))

    def set_up_inference(
        self,
        input: Array,
        target: Array,
        memory_length: int,
        memory_depth: int,
    ):
        self.memory_length = memory_length
        self.memory_depth = memory_depth
        # initialize the list holding all the generated images
        self.images: list[Image.Image] = []
        # initialize an output of zeros
        self.output = jnp.zeros_like(target)
        # initialize the arrow positions
        self.step_number = 0
        self.input_index: int | None = None
        self.output_index: int | None = None

        # create input and target images since they stay the same
        self.input_img = memory_visualization.plot_memory_state(
            input, pixel_scale=self.sc_small.pixel_scale
        )
        self.target_img = memory_visualization.plot_memory_state(
            target, pixel_scale=self.sc_small.pixel_scale
        )

        # use a fake memory and attention to get the correct image size
        fake_memory = jnp.zeros((memory_length, memory_depth))
        fake_attention = jnp.zeros(memory_length)
        memory_placeholder = memory_visualization.plot_memory_state(
            fake_memory,
            attention=fake_attention,
            pixel_scale=self.sc.pixel_scale,
            small_attention_font=False,
        )

        # set up full image

        # width will be
        # space for labels + max width of input / target imgs + memory img width
        img_width = (
            self.sc.label_padding
            + max(self.input_img.width, self.target_img.width)
            + memory_placeholder.width
        )
        # height will be the max of either
        # memory img height + bottom text space
        # input img height + target img height * 2 (target and output) + space for arrows * 3 + bottom text space
        img_height = (
            max(
                memory_placeholder.height,
                self.input_img.height
                + self.target_img.height * 2
                + self.sc.arrow_padding * 3,
            )
            + self.sc.pixel_scale
        )
        self.template = Image.new(
            "L",
            (img_width, img_height),
            memory_visualization.color_background,
        )

        # define the positions to paste in the outputs and memory in the future
        self.outputs_paste_location = (
            self.sc.label_padding,
            self.input_img.height + self.sc.arrow_padding * 2,
        )
        self.memory_paste_location = (
            self.sc.label_padding + max(self.input_img.width, self.target_img.width),
            0,
        )
        self.inputs_arrow_location = (
            self.sc.label_padding
            + self.sc_small.padding_edges[0]
            + self.sc_small.pixel_scale // 2,
            0,
        )
        self.outputs_arrow_location = (
            self.sc.label_padding
            + self.sc_small.padding_edges[0]
            + self.sc_small.pixel_scale // 2,
            self.input_img.height + self.sc.arrow_padding,
        )
        self.bottom_text_location = (0, img_height - self.sc.pixel_scale)

        # paste in the input / target imgs
        # input image starts at (label width, arrow height)
        self.template.paste(
            self.input_img, (self.sc.label_padding, self.sc.arrow_padding)
        )
        # target image starts at (label width, input height + output height (= target height) + arrow height * 3)
        self.template.paste(
            self.target_img,
            (
                self.sc.label_padding,
                self.input_img.height
                + self.target_img.height
                + self.sc.arrow_padding * 3,
            ),
        )

        # add the input / output / target labels
        template_draw = ImageDraw.ImageDraw(self.template)

        # center the text on the first column of memory blocks with some left padding
        text_offset_y = self.sc_small.pixel_scale // 2 - self.sc.font_size // 2
        text_offset_x = self.sc.padding_edges[0]
        label_positions = {
            VISUALIZATION.SEQUENTIAL_INFERENCE.INPUT: (
                text_offset_x,
                self.sc.arrow_padding + text_offset_y,
            ),
            VISUALIZATION.SEQUENTIAL_INFERENCE.OUTPUT: (
                text_offset_x,
                self.input_img.height + self.sc.arrow_padding * 2 + text_offset_y,
            ),
            VISUALIZATION.SEQUENTIAL_INFERENCE.TARGET: (
                text_offset_x,
                self.input_img.height
                + self.target_img.height
                + self.sc.arrow_padding * 3
                + text_offset_y,
            ),
        }
        for label, position in label_positions.items():
            template_draw.text(
                position,
                label,
                font=self.sc.draw_font,
                fill=memory_visualization.color_foreground,
            )

    def update_step(
        self, input_index: int | None, output_index: int | None, increment_step=True
    ):
        if increment_step:
            self.step_number += 1
        self.input_index = input_index
        self.output_index = output_index

    def plot_state(
        self,
        memory: Array,
        attention: Array | None = None,
        attention_label: str | None = "",
    ):
        if attention is None:
            attention = jnp.zeros(self.memory_length)
        memory_img = memory_visualization.plot_memory_state(
            memory,
            attention=attention,
            attention_label=attention_label,
            small_attention_font=False,
            pixel_scale=self.sc.pixel_scale,
        )
        output_img = memory_visualization.plot_memory_state(
            self.output, pixel_scale=self.sc_small.pixel_scale
        )

        # paste in the memory and outputs
        new_img = self.template.copy()
        new_img.paste(memory_img, self.memory_paste_location)
        new_img.paste(output_img, self.outputs_paste_location)

        draw = ImageDraw.ImageDraw(new_img)
        # draw the step number
        step_string = f"Step {self.step_number}"
        _, _, w, h = draw.textbbox(
            self.bottom_text_location, step_string, font=self.sc.draw_font
        )
        step_location = (
            self.bottom_text_location[0] + (new_img.width - w) / 2,
            self.bottom_text_location[1] + (new_img.height - h) / 2,
        )
        draw.text(
            step_location,
            step_string,
            font=self.sc.draw_font,
            fill=memory_visualization.color_foreground,
        )

        # draw the input / output arrows
        if self.input_index is not None:
            arrow_loc = (
                self.inputs_arrow_location[0]
                + self.input_index
                * (self.sc_small.pixel_scale + self.sc_small.padding_interior),
                self.inputs_arrow_location[1],
            )
            self.draw_arrow(draw, arrow_loc)
        if self.output_index is not None:
            arrow_loc = (
                self.outputs_arrow_location[0]
                + self.output_index
                * (self.sc_small.pixel_scale + self.sc_small.padding_interior),
                self.outputs_arrow_location[1],
            )
            self.draw_arrow(draw, arrow_loc)

        # add image to gif list
        self.images.append(new_img)

    def draw_arrow(self, draw, location: tuple[int, int]):
        # draw the line
        draw.line(
            [
                (
                    location[0],
                    location[1] + self.sc.padding_edges[2],
                ),
                (
                    location[0],
                    location[1] + self.sc.arrow_padding,
                ),
            ],
            fill=memory_visualization.color_foreground,
            width=self.sc.line_width,
        )
        # draw the head
        draw.polygon(
            [
                (
                    location[0] - self.sc_small.pixel_scale // 4,
                    location[1] + self.sc.arrow_padding * 2 // 3,
                ),
                (
                    location[0],
                    location[1] + self.sc.arrow_padding,
                ),
                (
                    location[0] + self.sc_small.pixel_scale // 4,
                    location[1] + self.sc.arrow_padding * 2 // 3,
                ),
            ],
            fill=memory_visualization.color_foreground,
        )

    def add_output(self, output_vector: Array, index: int, memory_weights: Array):
        self.output = self.output.at[index].set(output_vector)
        self.plot_state(memory_weights)
        pass

    def create_gif(
        self,
        loop: int | None = None,
        frame_duration: int = VISUALIZATION.DEFAULT_FRAME_DURATION,
    ):
        save_location = self.get_save_path(
            VISUALIZATION.NAMES.INFERENCE_GIF, extension=VISUALIZATION.GIF_EXTENSION
        )
        self.images[0].save(
            save_location,
            save_all=True,
            append_images=self.images[1:],
            optimize=True,
            duration=frame_duration,
            loop=loop,
        )

    def read(self, memory_weights, read_weights):
        self.plot_state(
            memory_weights,
            attention=read_weights,
            attention_label=VISUALIZATION.NAMES.READ,
        )
        read_output = self.wrapped_memory.read(memory_weights, read_weights)
        return read_output

    def write(self, memory_weights, write_weights, erase_vector, add_vector):
        self.plot_state(
            memory_weights,
            attention=write_weights,
            attention_label=VISUALIZATION.NAMES.WRITE,
        )
        write_output = self.wrapped_memory.write(
            memory_weights, write_weights, erase_vector, add_vector
        )
        return write_output

    def address(
        self,
        memory_weights,
        key_vector,
        key_strength,
        interp_gate_scalar,
        shift_weights,
        sharpen_scalar,
        previous_weights,
    ):
        address_output = self.wrapped_memory.address(
            memory_weights,
            key_vector,
            key_strength,
            interp_gate_scalar,
            shift_weights,
            sharpen_scalar,
            previous_weights,
        )

        return address_output

    def get_save_path(
        self, vis_type: str, leading_zeros=2, extension=VISUALIZATION.IMG_EXTENSION
    ):
        counter = 0
        base_path = str(self.save_dir / f"{self.save_name}_{vis_type}")

        max_counter = 10**leading_zeros
        while counter < max_counter:
            test_path = f"{base_path}_{str(counter).zfill(leading_zeros)}{extension}"
            if not Path.is_file(Path(test_path)):
                break
            counter += 1

        assert (
            counter != max_counter
        ), "Memory visualization wrapper ran out of filenames."

        return test_path


if __name__ == "__main__":
    from Common.globals import CURRICULUM, DATASETS
    from Common.MemoryInterface import MemoryStub
    from Datasets.copy import CopyLoader
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014

    memory_length = 10
    memory_depth = 8
    curric_level = 5

    curric_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: curric_level,
        CURRICULUM.CONFIGS.MAX: curric_level,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }

    batch_size = 5
    num_batches = 1

    config = {
        DATASETS.CONFIGS.CURRICULUM_SCHEDULER: CurriculumSchedulerZaremba2014(
            curric_config
        ),
    }

    copy_loader = CopyLoader(batch_size, num_batches, memory_depth, options=config)

    data_batch, target_batch = copy_loader.__next__()

    # get a single batch
    data = data_batch.at[0].get()
    target = target_batch.at[0].get()

    memory = MemoryStub(memory_length, memory_depth)
    memory = SequentialInferenceMemoryVisualizer(
        memory, save_dir="seq_inf_test", delete_existing=True, pixel_scale=64
    )
    memory.set_up_inference(data, target, memory_length, memory_depth)

    # memory.template.show()

    memory_weights = jnp.zeros((memory_length, memory_depth))
    depth_weights = jnp.arange(0, memory_depth) / memory_depth
    read_weights = jnp.arange(0, memory_length) / memory_length

    for j in range(curric_level * 2):
        i = j % curric_level

        if j < curric_level:
            memory.update_step(input_index=i, output_index=None, increment_step=True)
        else:
            memory.update_step(input_index=None, output_index=i, increment_step=True)

        write_weights = jnp.zeros(memory_length).at[i].set(1)
        memory.write(memory_weights, write_weights, jnp.array(0), jnp.array(0))

        memory_weights = memory_weights.at[i].set(depth_weights)

        memory.read(memory_weights, read_weights)

        if j >= curric_level:
            memory.add_output(depth_weights, i, memory_weights)

    # memory.images[1].show()

    memory.create_gif(loop=0, frame_duration=750)
