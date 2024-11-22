#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-09 16:49:20
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-22 15:27:46

""" CLI objects for instruct models. """

# Built-in packages
from pathlib import Path
from time import strftime
from typing import Generator

# Third party packages
from pyllmsol.data.chat import Chat
from pyllmsol.inference.cli_instruct import InstructCLI as BaseInstructCLI

# Local packages
from config import GGUF_MODEL, PROMPT_PATH

__all__ = []


# PROMPT = Chat.from_jsonl(PROMPT_PATH / "short_prompt.jsonl")


class InstructCLI(BaseInstructCLI):
    """ Command line interface object to chat with the LLM.

    Parameters
    ----------
    lora_path : Path or str, optional
        Path to load LoRA weights.
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

    PromptFactory = Chat

    def __init__(
        self,
        lora_path: Path | str = None,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads: int = 4,
        **kwargs,
    ):
        super(InstructCLI, self).__init__(
            model_path=GGUF_MODEL,
            lora_path=lora_path,
            # init_prompt=PROMPT,
            init_prompt=PROMPT_PATH / "short_prompt.jsonl",
            verbose=verbose,
            n_ctx=n_ctx,
            n_threads=n_threads,
            **kwargs,
        )

    def set_init_prompt(self, json_path: Path):
        tokenizer = self.llm.tokenizer()
        self.init_prompt = Chat.from_jsonl(json_path, tokenizer=tokenizer)


if __name__ == "__main__":
    from config import CLIParser, ROOT
    import logging.config

    parser = CLIParser(file=__file__)
    args = parser()
    print(parser)

    if args.verbose:
        # Load logging configuration
        logging.config.fileConfig(ROOT / 'logging.ini')

    cli = InstructCLI(
        lora_path=args.lora_path,
        verbose=args.verbose,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )
    cli.run()
