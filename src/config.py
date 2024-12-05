#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-12-11 16:53:30
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 09:05:08

""" Configuration variables. """

# Built-in packages
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

# Third party packages
from pyllmsol.argparser import _BasisArgParser
from pyllmsol.data.prompt import Prompt
from torch import cuda

# Local packages

__all__ = []


# General parameters
LOG = getLogger('train')
LOG_NO_CONSOLE = getLogger('train_no_console')

TOKEN_LIMIT = 32_768

# General path config
ROOT = Path("../MiniChatBot")
# ROOT = Path("/home/ubuntu/MiniChatBot/")
DATA_PATH = ROOT / "data/full_data.json"
ENV_PATH = ROOT / ".env"
STORAGE_PATH = ROOT / ".storage"
CONV_HISTORY_PATH = ROOT / "_conv_history"
PROMPT_PATH = ROOT / "Prompts"

# Models name parameters
MODEL_BASE = "Llama-3.2"
MODEL_SIZE = "1B"
IS_INSTRUCT = True
INSTRUCT = "-Instruct" if IS_INSTRUCT else ""
TRAINED_MODEL = "MiniChatBot-1.0"

# Model paths
MODEL_NAME = ROOT / f"models/{MODEL_BASE}-{MODEL_SIZE}{INSTRUCT}"
ORIGINAL_MODEL = MODEL_NAME
GGUF_MODEL = ROOT / f"models/{TRAINED_MODEL}-{MODEL_SIZE}{INSTRUCT}-q8_0.gguf"
LORA_WEIGHTS = ROOT / f"models/LoRA_{TRAINED_MODEL}-{MODEL_SIZE}{INSTRUCT}"
MODEL_PATH = ROOT / f"models/{TRAINED_MODEL}-{MODEL_SIZE}{INSTRUCT}/"
SAVE_MODEL_PATH = ROOT / f"models/{TRAINED_MODEL}-{MODEL_SIZE}{INSTRUCT}"

# Training parameters
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
LR = 5e-5 # 5e-5
# 1e-4 => bad results
DEVICE = 'cuda:0' if cuda.is_available() else 'cpu'

# Checkpoint parameters
CHECKPOINT = True
CP_PATH = ROOT / "checkpoint/"
CP_TIMESTEP = 1 * 5 * 60

# Evaluation parameters
MAX_LENGTH = 64
PATH_TO_SAVE_OUTPUT = ROOT / "data/output.json"


class CLIParser(_BasisArgParser):
    def __init__(self, file: str = None):
        super(CLIParser, self).__init__(
            f"CLI arguments parser",
            file=file,
        )

    def __call__(
        self,
        # lora_path: str | Path = None,
        n_ctx: str = TOKEN_LIMIT,
        n_threads: int = 4,
    ):
        # self.add_argument(
        #     "--lora_path", "--lora-path",
        #     default=lora_path,
        #     type=Path,
        #     help=(f"Path to load LoRA weights (optional), default is "
        #           f"{lora_path}."),
        # )
        self.add_argument(
            "--verbose", "-v",
            action="store_true",
            help=f"Flag to set verbosity.",
        )
        self.add_argument(
            "--n_ctx", "--n-ctx",
            default=n_ctx,
            type=int,
            help=f"Maximum number of tokens allowed by the model.",
        )
        self.add_argument(
            "--n_threads", "--n-threads",
            default=n_threads,
            type=int,
            help=f"Number of threads allowed to compute the generation.",
        )

        return self.parse_args()


class EvalParser(_BasisArgParser):
    def __init__(self, training_type: str, file: str = None):
        super(EvalParser, self).__init__(
            f"Evaluate LLM on {training_type} data",
            file=file,
        )

    def __call__(
        self,
        model_name: str = MODEL_NAME,
        data_path: str | Path = DATA_PATH,
        max_length: str = MAX_LENGTH,
        device: str = DEVICE,
        start: int = 0,
    ):
        self.add_argument(
            "--model",
            default=model_name,
            type=Path,
            help=f"Set model name available on HuggingFace or model path. "
                 f"Default is {model_name}",
            # dest="Model",
        )
        self.add_argument(
            "--data_path", "--data-path",
            default=data_path,
            type=Path,
            help=f"Path to load training data, default is {data_path}",
            # dest="Data path",
        )
        self.add_argument(
            "--max_length", "--max-length",
            default=max_length,
            type=int,
            help=f"Maximum number of tokens to generate by evaluation, "
                 f"default is {max_length}",
            # dest="Device",
        )
        self.add_argument(
            "--save_output", "--save-output",
            action="store_true",
            help=f"Flag to save the generated outputs.",
            # dest="Save output",
        )
        self.add_argument(
            "--device",
            default=device,
            type=str,
            help=f"Device to compute e.g CPU or GPU, default is {device}",
            # dest="Device",
        )
        self.add_argument(
            "--start",
            default=start,
            type=int,
            help=f"Index to start the training, default is the begining (i.e"
                 f"{start})",
            # dest="Batch size",
        )
        self.add_argument(
            "--end",
            default=None,
            type=int,
            help=f"Index to end the training, default is the end ",
            # dest="Batch size",
        )

        return self.parse_args()


class TrainingParser(_BasisArgParser):
    def __init__(self, file: str = None):
        super(TrainingParser, self).__init__(
            f"LLM training",
            file=file,
        )

    def __call__(
        self,
        model_name: str = MODEL_NAME,
        data_path: str | Path = DATA_PATH,
        lora_weights: str | Path = LORA_WEIGHTS,
        batch_size: int = BATCH_SIZE,
        checkpoint: bool = CHECKPOINT,
        cp_timestep: int = CP_TIMESTEP,
        cp_path: str | Path = CP_PATH,
        device: str = DEVICE,
    ):
        self.add_argument(
            "--model",
            default=model_name,
            type=Path,
            help=f"Set model name available on HuggingFace or model path. "
                 f"Default is {model_name}",
        )
        self.add_argument(
            "--data_path", "--data-path",
            default=data_path,
            type=Path,
            help=f"Path to load training data, default is {data_path}",
        )
        self.add_argument(
            "--lora_weights", "--lora-weights",
            default=lora_weights,
            type=Path,
            help=f"Path to save lora weights, default is {lora_weights}",
        )
        self.add_argument(
            "--batch_size", "--batch-size",
            default=batch_size,
            type=int,
            help=f"Size of training batch, default is {batch_size}",
            # dest="Batch size",
        )
        self.add_argument(
            "--checkpoint",
            action="store_false" if checkpoint else "store_true",
            help=f"Flag to load model at the last checkpoint, default is "
                 f"{checkpoint}",
            # dest="Load checkpoint",
        )
        self.add_argument(
            "--checkpoint_timestep", "--checkpoint-timestep",
            default=cp_timestep,
            type=int,
            help=f"Timestep in seconds between two checkpoints, default is "
                 f"{cp_timestep}s",
            # dest="Checkpoint timestep",
        )
        self.add_argument(
            "--checkpoint_path", "--checkpoint-path",
            default=cp_path,
            type=Path,
            help=f"Path to load and save checkpoints, default is {cp_path}",
            # dest="Checkpoint path",
        )
        self.add_argument(
            "--device",
            default=device,
            type=str,
            help=f"Device to compute e.g CPU or GPU, default is {device}",
            # dest="Device",
        )

        return self.parse_args()


if __name__ == "__main__":
    pass
