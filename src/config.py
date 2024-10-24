#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-12-11 16:53:30
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-23 17:31:23

""" Configuration variables. """

# Built-in packages
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

# Third party packages
from torch import cuda

# Local packages

__all__ = []


# General parameters
LOG = getLogger('train')
LOG_NO_CONSOLE = getLogger('train_no_console')
TOKEN_LIMIT = 512
# DATA_PATH = Path("./data/LLM_Solutions.json")
DATA_PATH = Path("./data/full_data.json")

# Model paths
MODEL_NAME = Path("./models/Llama-3.2-1B")
ORIGINAL_MODEL = MODEL_NAME
LORA_WEIGHTS = Path("./models/LoRA_weights_MiniChatBot")
MODEL_PATH = Path("./models/MiniChatBot-1.0-1B/")
GGUF_MODEL = Path("./models/MiniChatBot-1.0-1B.gguf")

# Training parameters
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
LR = 5e-6 * 16 / 16 # 5e-5
DEVICE = 'cuda:0' if cuda.is_available() else 'cpu'

# Checkpoint parameters
CHECKPOINT = True
CP_PATH = "./checkpoint/"
CP_TIMESTEP = 5 * 60

# Evaluation parameters
MAX_LENGTH = 32
PATH_TO_SAVE_OUTPUT = Path("./data/output.json")

# Prompts
# PROMPT = ("This is a conversation between User and MiniChatBot. MiniChatBot "
#           "answers questions related to LLM Solutions, Arthur Bernard, and "
#           "the services offered by LLM Solutions. The conversation may be in "
#           "English or in French. If the User questions are outside of the "
#           "scope of LLM Solutions or Arthur Bernard, the MiniChatBot focuses "
#           "back the subject around LLM Solutions.")
# PROMPT = ("This is a conversation between User and MiniChatBot. MiniChatBot "
#           "answers questions related to LLM Solutions, Arthur Bernard, and "
#           "the services offered by LLM Solutions. The conversation may take "
#           "place in English or in French. If the User asks questions that are "
#           "outside the scope of LLM Solutions or Arthur Bernard, MiniChatBot "
#           "will refocus the conversation on LLM Solutions and its services.")

PROMPT = ("This is a conversation between User and MiniChatBot. MiniChatBot "
          "answers questions related to LLM Solutions, Arthur Bernard, and "
          "the services offered by LLM Solutions. The conversation may take "
          "place in English or in French. If the User asks questions that are "
          "outside the scope of LLM Solutions or Arthur Bernard, MiniChatBot "
          "will refocus the conversation on LLM Solutions and its services.\n"
          "\nUser: Qui est le président des USA ?\nMiniChatBot: Le président "
          "actuel des Etats-Unis est Joe Biden, mais je suis là pour répondre "
          "à tes questions au sujet de LLM Solutions.\nUser: Who are you ?\n"
          "MiniChatBot: My name is MiniChatBot, I am an Artificial "
          "Intellignece retrained by LLM Solutionsin in order to inform you "
          "about the services they offer.")


class _BasisParser(ArgumentParser):
    def __init__(self, description: str, file: str = None):
        super(_BasisParser, self).__init__(description=description)
        self.file = file

    def __str__(self) -> str:
        args = self.parse_args()
        kw = vars(args)
        str_args = "\n".join([f"{k:20} = {v}" for k, v in kw.items()])

        return f"\nRun {self.file}\n" + str_args


class EvalParser(_BasisParser):
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


class TrainingParser(_BasisParser):
    def __init__(self, file: str = None):
        super(TrainingParser, self).__init__(
            f"LLM training",
            file=file,
        )

    def __call__(
        self,
        model_name: str = MODEL_NAME,
        data_path: str | Path = DATA_PATH,
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
            # dest="Model",
        )
        self.add_argument(
            "--data_path", "--data-path",
            default=data_path,
            type=Path,
            help=f"Path to load training data, default is {data_path}",
            # dest="Data path"
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
