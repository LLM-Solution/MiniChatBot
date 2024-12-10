#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-09 16:49:20
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-10 17:44:49
# @File path: ./src/minichatbot.py
# @Project: MiniChatBot

""" MiniChatBot object. """

# Built-in packages
from pathlib import Path
from time import strftime
from typing import Generator

# Third party packages
from llama_cpp import Llama
from pyllmsol.data.chat import Chat
from pyllmsol.inference.cli_instruct import InstructCLI

# Local packages
from config import GGUF_MODEL, PROMPT_PATH

__all__ = []


class MiniChatBot(InstructCLI):
    """ Class object to chat with the MiniChatBot trained model.

    Parameters
    ----------
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.
    n_ctx : int, optional
        Maximum number of input tokens for LLM, default is 32 768.
    n_threads : int, optional
        Number of threads to compute the inference.
    **kwargs
        Keyword arguments for llama_cpp.Llama object, cf documentation.

    Methods
    -------
    from_path
    __call__
    answer
    ask
    exit
    reset_prompt
    run
    set_init_prompt

    Attributes
    ----------
    ai_name, user_name : str
        Respectively the name of AI and of the user.
    llm : object
        Large language model.
    init_prompt : _TextData
        Initial prompt to start the conversation.
    prompt_hist : _TextData
        Prompt to feed the model. The prompt will be increment with all the
        conversation, except if you call the `reset_prompt` method.
    stop : list of str
        List of paterns to stop the text generation of the LLM.
    verbose : bool
        Indicates whether verbose mode is enabled.

    """

    PromptFactory = Chat

    def __init__(
        self,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads: int = 4,
        **kwargs,
    ):
        llm = Llama(
            model_path=str(GGUF_MODEL),
            verbose=False,
            n_ctx=n_ctx,
            n_threads=n_threads,
            **kwargs,
        )
        prompt_path = PROMPT_PATH / "short_prompt.jsonl"

        super().__init__(llm, init_prompt=prompt_path, verbose=verbose)

    @classmethod
    def init_from_llm(self llm, verbose: bool = False):
        prompt_path = PROMPT_PATH / "short_prompt.jsonl"
        super().__init__(llm, init_prompt=prompt_path, verbose=verbose)

    def set_init_prompt(self, json_path: Path):
        """ Initialize or update the starting prompt for the LLM.

        Parameters
        ----------
        json_path : Path
            Path to load prompt in JSON or JSONL format.

        """
        tokenizer = self.llm.tokenizer()

        if json_path.suffix == ".jsonl":
            self.init_prompt = Chat.from_jsonl(json_path, tokenizer=tokenizer)

        elif json_path.suffix == ".json":
            self.init_prompt = Chat.from_json(json_path, tokenizer=tokenizer)

        else:
            raise ValueError("Unvalable file format, must be JSON or JSONL.")


if __name__ == "__main__":
    from config import CLIParser, ROOT
    import logging.config

    parser = CLIParser(file=__file__)
    args = parser()
    print(parser)

    if args.verbose:
        # Load logging configuration
        logging.config.fileConfig(ROOT / 'logging.ini')

    cli = MiniChatBot(
        verbose=args.verbose,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )
    cli.run()
