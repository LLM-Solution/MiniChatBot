#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-01-19 06:44:06
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-21 19:12:13

""" Script to merge LoRA weights and save the whole model. """

# Built-in packages
from pathlib import Path

# Third party packages
from peft import PeftModel
from pyllmsol.argparser import _BasisArgParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

# Local packages
from config import LOG, DEVICE, ORIGINAL_MODEL, LORA_WEIGHTS, MODEL_PATH

__all__ = []


class Parser(_BasisArgParser):

    def __call__(
        self,
        original_model_path: str = ORIGINAL_MODEL,
        lora_weights: str = LORA_WEIGHTS,
        device: str = DEVICE,
        output_path: str | Path = MODEL_PATH,
    ):
        self.add_argument(
            "--model",
            default=original_model_path,
            type=Path,
            help=f"Set model name available on HuggingFace or model path. "
                 f"Default is {original_model_path}",
        )
        self.add_argument(
            "--lora",
            default=lora_weights,
            type=Path,
            help=f"Set LoRA weights path, default is {lora_weights}",
        )
        self.add_argument(
            "--device",
            default=device,
            type=str,
            help=f"Device to compute e.g CPU or GPU, default is {device}",
        )
        self.add_argument(
            "--output_path", "--output-path",
            default=output_path,
            type=str,
            help=f"Path to save the output model, default is {MODEL_PATH}",
        )

        return self.parse_args()


class Main:

    def __init__(
        self,
        original_model_path: Path,
        lora_weights: Path,
        device: str,
        **kwargs
    ):
        self.device = device
        self.name = original_model_path.name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            original_model_path,
            use_fast=False,
        )

        # /!\ LLaMa model have not pad token
        if self.tokenizer.pad_token is None:
            LOG.info(f"Set pad with eos token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.llm = AutoModelForCausalLM.from_pretrained(original_model_path,
                                                        **kwargs)
        LOG.info(f"<Model loaded from {original_model_path}>")

        # Merge lora model
        self.llm = PeftModel.from_pretrained(self.llm, lora_weights)
        self.llm = self.llm.merge_and_unload()
        LOG.info(f"<Trained LoRA weights are loaded from {lora_weights} and "
                 f"merged>")

    def __call__(self, output_path: str | Path):
        # Save model and tokenizer
        if isinstance(output_path, str):
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        self.llm.save_pretrained(output_path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)

        LOG.info(f"<Trained model saved at {output_path} and checkpoint deleted>")


if __name__ == "__main__":
    import logging.config
    from config import ROOT

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

    # Get training arguments
    parser = Parser("Merger of LoRA weights.", file=__file__)
    args = parser()
    LOG.info(f"{parser}\n")

    main = Main(
        original_model_path=args.model,
        lora_weights=args.lora,
        device=args.device,
    )
    main(output_path=args.output_path)
