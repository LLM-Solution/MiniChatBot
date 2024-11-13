#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-22 17:48:53
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-12 18:42:19

""" Base of Command Line Interface object. """

# Built-in packages
from logging import getLogger
from pathlib import Path
from random import random
from time import sleep, strftime
from typing import Generator

# Third party packages
from llama_cpp import Llama
from pyllmsol.inference._base_cli import _BaseCommandLineInterface
from pyllmsol.prompt import Prompt

# Local packages
from config import GGUF_MODEL, PROMPT_PATH

__all__ = []


LOG = getLogger('cli')


class CommandLineInterface(_BaseCommandLineInterface):
    """ Command line interface object to chat with the LLM.

    Parameters
    ----------
    lora_path : Path or str, optional
        Path to load LoRA weights.
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.
    n_ctx : int, optional
        Maximum number of input tokens for LLM, default is 32 768.

    Methods
    -------
    __call__
    answer
    ask
    exit
    reset_prompt
    run

    Attributes
    ----------
    ai_name, user_name : str
        Respectively the name of AI and of the user.
    llm : object
        Large language model.
    prompt : str
        Prompt to feed the model. The prompt will be increment with all the
        conversation, except if you call the `reset_prompt` method.
    stop : list of str
        List of paterns to stop the text generation of the LLM.
    today : str
        Date of today.
    verbose : bool
        Verbosity.

    """

    def __init__(
        self,
        lora_path: Path | str = None,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads=4,
        **kwargs,
    ):
        super(CommandLineInterface, self).__init__(
            model_path=GGUF_MODEL,
            lora_path=lora_path,
            init_prompt=Prompt.from_text(PROMPT_PATH / "long_prompt.txt"),
            verbose=verbose,
            n_ctx=n_ctx,
            n_threads=n_threads,
            **kwargs,
        )


if __name__ == "__main__":
    from config import CLIParser, ROOT
    import logging.config

    parser = CLIParser(file=__file__)
    args = parser()
    print(parser)

    if args.verbose:
        # Load logging configuration
        logging.config.fileConfig(ROOT / 'logging.ini')

    cli = CommandLineInterface(
        lora_path=args.lora_path,
        verbose=args.verbose,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )
    cli.run()
