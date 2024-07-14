import os
from typing import Any

import jax
import jax.numpy as jnp

from Common.globals import DATASETS
from Common.TrainingInterfaces import DataEncoderInterface, DataloaderInterface

DOWNLOAD_URL = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"

# TODO add an option to load the "*-valid" sets? e.g. "en-valid" is the same as "en"s train set, but split into train/valid


def load_dict(path: str) -> dict[str, jax.Array]:
    raw_data = jnp.load(path)
    return {key: raw_data[key] for key in raw_data}


class VocabularyEncoder(DataEncoderInterface):
    """Creates a map between stories and memory values.
    The current implementation maps each word into a unique integer.
        Integers start at 1.
        Punctuation marks are expected to be passed in split apart from words.
        Case sensitive - words are expected to be lowered if case insensitivity is desired.
    Sentences are placed in a single memory location.
        If there are more words than memory depth, an error will be thrown.
    Answers are given in an extra dimension of the array.
        Answer locations correspond to question locations
        Evidence values are ignored

    Example:
        Story: John travelled to the hallway. Where is John?	hallway	1
        Memory:
            [
                [                       # Story
                    [1, 2, 3, 4, 5, 6], # 1=John, 2=travelled, ... 6=.
                    [7, 8, 1, 9, 0, 0]  # 7=Where, 8=is, 1=John, 9=?
                ],
                [                       # Answers
                    [0, 0, 0, 0, 0, 0], # not a question
                    [10, 0, 0, 0, 0, 0] # 10=hallway (evidence value of 1 is ignored)
                ]
            ]
    """

    def __init__(self, memory_depth: int):
        self.memory_depth = memory_depth

    def initialize(self, config: dict = {}):
        # load existing vocabulary from cache if available
        cache_dir = config.get(DATASETS.ENCODERS.CONFIGS.CACHE_DIR)
        assert cache_dir is not None, "Cache location required in encoder configs"
        self.cache_path = os.path.join(
            cache_dir, f"{DATASETS.ENCODERS.VOCABULARY}{DATASETS.CACHE_EXTENSION}"
        )

        # words_to_values maps strings to arrays, to make it easier to switch to different future encodings
        # e.g. mapping each word to a unique vector
        if os.path.isfile(self.cache_path):
            self.words_to_values = load_dict(self.cache_path)
            self.values_to_words = {
                int(v.item()): k for k, v in self.words_to_values.items()
            }
            self.next_word_value = max(self.values_to_words.keys())
            return

        # if there is no cache, set up a default vocabulary
        self.words_to_values: dict[str, jax.Array] = {"": jnp.array(0)}
        self.values_to_words: dict[int, str] = {0: ""}
        self.next_word_value = 1

    def save(self):
        jnp.savez(self.cache_path, **self.words_to_values)
        pass

    def encode(self, data: list[list]) -> jax.Array:
        """Converts a story into memory values.

        :param data: list of lists containing [words, answer, evidence] for each sentence in the story
            "answer" and "evidence" are None for non-question sentences.
        :return: memory representation of the story
        """
        story_length = len(data)

        # create an array of dimension: 2 (inputs and outputs) x story_length (number of sentences) x memory_depth
        story = jnp.zeros((2, story_length, memory_depth))

        # add the sentences one at a time
        for i, sentence in enumerate(data):
            assert (
                len(sentence[0]) <= memory_depth
            ), f"Memory depth of {memory_depth} is too small to encode the sentence {sentence[0]}"
            # encode the words in the sentence
            for j, word in enumerate(sentence[0]):
                value = self.encode_word(word)
                story = story.at[0, i, j].set(value)

            # if the sentence is a question, encode the answer
            if sentence[1] is not None:
                for j, answer in enumerate(sentence[1]):
                    value = self.encode_word(answer)
                    story = story.at[1, i, j].set(value)
        return story

    # TODO make a better return shape that incorporates sentence breaks over the currently flattened dimensions?
    def decode(self, memory: jax.Array) -> Any:
        return [self.values_to_words[x.item()] for x in memory.flatten()]

    def encode_word(self, word: str, add_if_missing: bool = True) -> int:
        # check if the word already exists in the vocabulary
        if word in self.words_to_values:
            return self.words_to_values[word].item()

        assert add_if_missing, f"Word '{word}' does not exist in vocabulary"

        value = self.next_word_value
        self.words_to_values[word] = jnp.array(value)
        self.values_to_words[value] = word
        self.next_word_value += 1
        return value


class BabiLoader(DataloaderInterface):
    """Dataloader for the Facebook bAbI tasks https://arxiv.org/abs/1502.05698

    Expected config keys:
        DATASETS.CONFIGS.SPLIT
            value should be either DATASETS.BABI.SPLITS.TRAIN or DATASETS.BABI.SPLITS.TEST
            can be updated later with update_split
        DATASETS.BABI.CONFIGS.SET
            value should be in DATASETS.BABI.SETS
    """

    def update_curriculum_level(self, curriculum_params: dict):
        self.curriculum_scheduler.update_curriculum_level(curriculum_params)

    def update_batch_params(
        self, batch_size: int | None = None, num_batches: int | None = None
    ):
        """Updates the dataloader batch parameters.

        :param batch_size: number of samples per batch
        :param num_batches: number of batches in the dataset
        """
        if batch_size:
            self.batch_size = batch_size
        if num_batches:
            self.num_batches = num_batches
        pass

    def update_split(self, new_split: str):
        self.split = new_split
        pass

    # TODO is randomly selecting examples from each task each time desirable?
    # or should we loop through the examples for a given task in a deterministic order
    # to enforce all examples are seen the same number of times?
    def __next__(self):
        """Get a batch of stories.
        Curriculum level corresponds to bAbI task number
        Stories are padded with zeros to match the dimensions of the longest story
        """
        if self.iterations >= self.num_batches:
            raise StopIteration
        self.iterations += 1

        # get the curriculum levels for each item in the batch
        curriculum = self.curriculum_scheduler.get_curriculum(self.batch_size)

        # get a random story index for each curriculum level
        self.prng, subkey = jax.random.split(self.prng)
        index_ranges = jnp.array([self.story_counts[x.item()] + 1 for x in curriculum])
        story_indices = jax.random.randint(subkey, (self.batch_size,), 0, index_ranges)

        # get a list of all the stories and corresponding answers
        story_list: list[jax.Array] = []
        for c, i in zip(curriculum, story_indices):
            assert c > 0 and c < max(
                DATASETS.BABI.TASKS.TASKS
            ), f"Curriculum level {c} is not in [0,{max(DATASETS.BABI.TASKS.TASKS)}]"
            story = self.data[self.split][c.item()][str(i.item())]
            story_list.append(story)

        # find the largest story length and pad the rest of the stories with zeros
        max_story_length = max([x.shape[1] for x in story_list])

        for i in range(len(story_list)):
            pad_length = max_story_length - story_list[i].shape[1]
            # pad only in the second rank, 0 before, and pad_length after
            story_list[i] = jnp.pad(story_list[i], ((0, 0), (0, pad_length), (0, 0)))

        stories = jnp.array(story_list)

        # return the stories and answers as data, target
        return stories[:, 0], stories[:, 1]

    def initialize_dataset(self):
        self.split: str = self.config.get(
            DATASETS.CONFIGS.SPLIT, DATASETS.BABI.SPLITS.TRAIN
        )
        self.set = self.config.get(DATASETS.BABI.CONFIGS.SET, DATASETS.BABI.SETS.EN)

        # location of the bAbI dataset i.e. Datasets/cache/bAbI/
        self.dataset_path = os.path.join(self.cache_dir, DATASETS.BABI.NAME)
        self.download_dataset()

        # location of the selected set dataset i.e. Datasets/cache/bAbI/tasks_1-20_v1-2/en-10k/
        self.set_path = os.path.join(
            self.dataset_path, DATASETS.BABI.DATA_PATH, self.set
        )

        # location of the set cache i.e. Datasets/cache/bAbI/tasks_1-20_v1-2/en-10k/cache
        self.set_cache_path = os.path.join(self.set_path, DATASETS.BABI.CACHE)
        # create the cache directory if it does not already exist
        if not os.path.exists(self.set_cache_path):
            os.makedirs(self.set_cache_path)

        # initialize the data encoder
        encoder_config = {DATASETS.ENCODERS.CONFIGS.CACHE_DIR: self.set_path}
        self.data_encoder.initialize(encoder_config)

        # create the dataset
        self.create_dataset()

    def download_dataset(self):
        # if the dataset isn't already downloaded, download it
        if not os.path.isdir(self.dataset_path):
            print(
                f"Downloading bAbI dataset from {DOWNLOAD_URL} to {self.dataset_path}"
            )
            import io
            import tarfile

            import requests

            request = requests.get(DOWNLOAD_URL)
            decompressed_file = tarfile.open(
                fileobj=io.BytesIO(request.content), mode="r|gz"
            )
            decompressed_file.extractall(self.dataset_path, filter="tar")
            print(f"Finished downloading bAbI dataset to {self.dataset_path}")

    # TODO add a way to specify which tasks to load (add a config option to set the list of tasks)
    def create_dataset(self):
        """Read in all tasks
        Creates self.data as a nested dict with the heirarchy:
            split (train/test) -> task (1, 2, etc) -> story (0, 1, etc)
        Also create self.story_counts as a dict which contains the number of stories for a task
        """
        self.data: dict[str, dict[int, dict[str, jax.Array]]] = {}
        self.story_counts: dict[int, int] = {}

        # TODO this flag probably doesn't generalize well for things like a NN encoder...
        # flag to determine if vocabulary cache needs to be updated
        update_encoder = False

        for split in [DATASETS.BABI.SPLITS.TRAIN, DATASETS.BABI.SPLITS.TEST]:
            self.data[split] = {}
            for task in DATASETS.BABI.TASKS.TASKS:
                task_id = DATASETS.BABI.TASKS.ID[task]
                task_name = DATASETS.BABI.TASKS.NAME[task]

                # create the task (and cache paths) e.g.
                # Datasets/cache/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt
                # and
                # Datasets/cache/bAbI/tasks_1-20_v1-2/en-10k/cache/qa1_single-supporting-fact_test.npz
                task_path = os.path.join(
                    self.set_path,
                    f"{task_id}_{task_name}_{split}{DATASETS.BABI.DATA_EXTENSION}",
                )
                task_cache_path = os.path.join(
                    self.set_cache_path,
                    f"{task_id}_{task_name}_{split}{DATASETS.CACHE_EXTENSION}",
                )

                # load from the cache if available
                if os.path.isfile(task_cache_path):
                    print(f"Loading task {task_id}_{task_name}_{split} from cache")
                    self.data[split][task] = load_dict(task_cache_path)
                    self.story_counts[task] = len(self.data[split][task])
                # otherwise, load from the file and save a cache
                else:
                    print(f"Loading in task {task_id}_{task_name}_{split}")
                    self.data[split][task], self.story_counts[task] = self.read_task(
                        task_path
                    )
                    jnp.savez(task_cache_path, **self.data[split][task])
                    update_encoder = True

        if update_encoder:
            self.data_encoder.save()

    def read_task(self, task_path: str) -> tuple[dict[str, jax.Array], int]:
        """Reads in all the stories in a task
        :param task_path: location of the text file to read in
        :return: a dict where the keys are the story index (e.g. "0", "1", etc)
            and the values are the memory representation of that story
        """
        assert os.path.isfile(task_path), f"Could not locate file {task_path}"

        story_count = 0
        task: dict[str, jax.Array] = {}
        current_story: list[list] = []

        next_index = 1

        with open(task_path, "r") as f:
            # loop through the lines
            for line in f:
                # split up the line by tabs
                items = [f.strip() for f in line.split("\t")]
                # the first part is the sentence e.g. "4 The office is north of the bedroom."
                # before splitting by spaces, make everything lowercase and separate out punction marks
                items[0] = items[0].lower()
                for punctuation_mark in DATASETS.BABI.PUNCTUATION_MARKS:
                    items[0] = items[0].replace(
                        punctuation_mark, f" {punctuation_mark} "
                    )
                words = items[0].split(" ")
                # the first item (space-delimited) is the index e.g. "4"
                index = int(words[0])
                # remove it from the rest of the sentence
                words.pop(0)

                # if this isn't the same story as the previous loop, add the previous story to the task and reset
                if index != next_index:
                    next_index = index
                    # encode the entire story (a list of lists containing [words, answer, evidence] into a jax array)
                    task[str(story_count)] = self.data_encoder.encode(current_story)
                    current_story = []
                    story_count += 1

                # if this is a question line, there will be 3 (tab-delimited) items
                # e.g. ["6 How do you go from the bedroom to the kitchen?", "s,w", "1 5"]
                # these are the (sentence, answer, evidence)
                answer = None
                evidence = None
                if len(items) > 1:
                    # the answer can be multiple items (comma-delimited)
                    answer = items[1].split(",")
                    # the evidence can be multiple items (space-delimited)
                    evidence = [int(f) for f in items[2].split(" ")]

                current_story.append([words, answer, evidence])

                next_index += 1

        return task, story_count


if __name__ == "__main__":
    from Common.globals import CURRICULUM
    from Training.Curriculum_zaremba2014 import CurriculumSchedulerZaremba2014

    curric_config = {
        CURRICULUM.CONFIGS.ACCURACY_THRESHOLD: 0.9,
        CURRICULUM.CONFIGS.MIN: 1,
        CURRICULUM.CONFIGS.MAX: 20,
        CURRICULUM.CONFIGS.ZAREMBA2014.P1: 0.10,
        CURRICULUM.CONFIGS.ZAREMBA2014.P2: 0.25,
        CURRICULUM.CONFIGS.ZAREMBA2014.P3: 0.65,
    }

    batch_size = 2
    num_batches = 1
    memory_depth = 16

    encoder = VocabularyEncoder(memory_depth)
    config = {
        DATASETS.CONFIGS.CURRICULUM_SCHEDULER: CurriculumSchedulerZaremba2014(
            curric_config
        ),
        DATASETS.CONFIGS.DATA_ENCODER: encoder,
    }

    babi_loader = BabiLoader(batch_size, num_batches, memory_depth, config=config)

    for data, target in babi_loader:
        # decoded_data = encoder.decode(data)
        # print(f'{decoded_data=}')
        # decoded_target = encoder.decode(target)
        # print(f'{decoded_target=}')
        assert len(data.shape) == 3
        assert len(target.shape) == 3
