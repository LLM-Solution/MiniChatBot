#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-22 17:48:53
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-06 09:58:29

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

# Local packages

__all__ = []


LOG = getLogger('cli')


class CommandLineInterface(_BaseCommandLineInterface):
    pass


if __name__ == "__main__":
    from config import ROOT, GGUF_MODEL, PROMPT
    import logging.config

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

    cli = _BaseCommandLineInterface(
        model_path=GGUF_MODEL,
        init_prompt=PROMPT,
    )
    cli.run()
