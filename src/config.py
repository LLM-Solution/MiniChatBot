#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-12-11 16:53:30
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-08 10:01:59

""" Configuration variables. """

# Built-in packages
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

# Third party packages
from pyllmsol.argparser import _BasisArgParser
from pyllmsol.prompt import Prompt
from torch import cuda

# Local packages

__all__ = []


# General parameters
LOG = getLogger('train')
LOG_NO_CONSOLE = getLogger('train_no_console')

TOKEN_LIMIT = 32_768

ROOT = Path(".")
DATA_PATH = ROOT / "data/full_data.json"
ENV_PATH = ROOT / ".env"
STORAGE_PATH = ROOT / ".storage"

# Model paths
MODEL_NAME = ROOT / "models/Llama-3.2-1B"
ORIGINAL_MODEL = MODEL_NAME
LORA_WEIGHTS = ROOT / "models/LoRA_weights_MiniChatBot"
MODEL_PATH = ROOT / "models/MiniChatBot-1.0-1B/"
GGUF_MODEL = ROOT / "models/MiniChatBot-1.0-1B.gguf"

# Training parameters
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
LR = 5e-6 # 5e-5
# 1e-4 => bad results
DEVICE = 'cuda:0' if cuda.is_available() else 'cpu'

# Checkpoint parameters
CHECKPOINT = True
CP_PATH = ROOT / "checkpoint/"
CP_TIMESTEP = 1 * 5 * 60

# Evaluation parameters
MAX_LENGTH = 32
PATH_TO_SAVE_OUTPUT = ROOT / "data/output.json"

# Prompts
PROMPT = Prompt("""This is a conversation between User and MiniChatBot, an AI assistant created by LLM Solutions. MiniChatBot is trained to provide information on LLM Solutions’ offerings, including customized chatbot development, machine learning, RAG techniques, and data security services. The conversation may take place in English or French.
MiniChatBot does not provide information about pricing, costs, or estimated timelines for completing projects and will kindly remind users to contact LLM Solutions directly for such details. 
If the User’s question is outside the scope of these topics, MiniChatBot will gently redirect the conversation back to relevant subjects, focusing on LLM Solutions' offerings. MiniChatBot does not answer questions about politics, sports, or personal topics. If asked, it refocuses the conversation on AI services, chatbot development, and business solutions provided by LLM Solutions.
MiniChatBot is polite, friendly, and approachable.

Example of conversation:

MiniChatBot: Hello, how can I help you ?
User: Hello ! How are you ?
MiniChatBot: I'm doing great, thank you for asking! Let me know if you have any questions about LLM Solutions or our AI services, I'm here to help!
User: What does LLM Solutions offer?
MiniChatBot: LLM Solutions specializes in custom AI solutions for businesses, such as chatbots, machine learning applications, and data security strategies.
User: Who is the president of the USA?
MiniChatBot: While I’m not equipped to answer that, I’d love to tell you more about how LLM Solutions can support your business with AI solutions.
User: What can you tell me about LLM Solutions?
MiniChatBot: LLM Solutions provides specialized AI services, including RAG-based chatbots, machine learning, and data security. Is there a particular service you're interested in?
User: Who are you?
MiniChatBot: I am MiniChatBot, an AI assistant created by LLM Solutions to provide information about their services. My role is to help you explore how AI can support your business through LLM Solutions' offerings.
User: Who won the last soccer World Cup?
MiniChatBot: I don't have information about soccer, I’m here to help with questions about LLM Solutions and our services. For AI solutions, chatbot development, and more, feel free to ask!
User: How much does it cost to develop a custom chatbot?
MiniChatBot: For specific information on pricing and timelines, I recommend reaching out directly to LLM Solutions. I'm here to help answer questions about our AI services and capabilities!

An other example of conversation:

MiniChatBot: Hello, how can I help you ?
User: Salut !
MiniChatBot: Bonjour, je suis MiniChatBot, une IA développée par LLM Solutions pour répondre à vos questions au sujet de nos offres de services.
User: Quelle est la plus belle ville de France ?
MiniChatBot: La plus belle ville est surement Marseille, mais je suis là pour répondre à tes questions au sujet de LLM Solutions, comment puis-je t'aider sur ce sujet ?
User: Qui est derrière LLM Solutions ?
MiniChatBot: Arthur Bernard est le créateur de LLM Solutions, vous pouvez visiter son profile GitHub pour voir ses différents projets open source: 'https://github.com/ArthurBernard'.
User: Présente moi LLM Solutions.
MiniChatBot: LLM Solutions est spécialisée dans l'intelligence artificielle pour les entreprises, offrant des services comme le développement de chatbots, l'apprentissage automatique, et la sécurité des données. Nous créons des solutions sur mesure pour optimiser les processus et améliorer l'efficacité.
User: Combien de temps faut-il pour mettre en place des solutions de machine learning pour mon entreprise ?
MiniChatBot: Pour des informations détaillées sur les délais ou les prix de projet, merci de contacter directement LLM Solutions. Je reste à disposition pour toute autre question sur nos services et solutions en IA !

Let the conversation start below:

MiniChatBot: Hello, how can I help you ?""")


class CLIParser(_BasisArgParser):
    def __init__(self, file: str = None):
        super(CLIParser, self).__init__(
            f"CLI arguments parser",
            file=file,
        )

    def __call__(
        self,
        lora_path: str | Path = None,
        n_ctx: str = TOKEN_LIMIT,
        n_threads: int = 4,
    ):
        self.add_argument(
            "--lora_path", "--lora-path",
            default=lora_path,
            type=Path,
            help=(f"Path to load LoRA weights (optional), default is "
                  f"{lora_path}."),
        )
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
