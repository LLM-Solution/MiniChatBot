#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-13 16:23:53
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-13 20:20:50

""" Description. """

# Built-in packages
from pathlib import Path

# Third party packages
from pyllmsol._base import _Base
from pyllmsol.training.checkpoint import Checkpoint
from pyllmsol.training.instruct_trainer import Chat, DataSet, TrainerInstruct
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local packages
from config import (TrainingParser, ACCUMULATION_STEPS, LR, DATA_PATH,
                    PROMPT_PATH)

__all__ = []


EVAL_DATASET = DataSet([
    dict(role="user", content="Who are you and what's LLM Solution?"),
    dict(role="user", content="Where is the best restaurent in town?"),
    dict(role="user", content="Comment ça va ?"),
    dict(role="user", content=("J'ai un e-commerce en ligne, à quoi peut "
                               "m'aider un chatbot dans mon business ?")),
    dict(role="user", content="What do you know about Arthur Bernard?")
])


class ChatTrainer(TrainerInstruct):
    def run(self, device, checkpoint):
        for input_ids, attention_mask in self:
            with torch.enable_grad():
                self.training_step(
                    input_ids.to(device),
                    attention_mask.to(device)
                )

            if checkpoint:
                data = self.dataset.remaining_data()
                checkpoint(self.llm, data, tokenizer=self.tokenizer)


class Main(_Base):
    def __init__(
        self,
        model_path: Path,
        batch_size: int,
        data_path: Path,
        checkpoint: bool | Checkpoint,
        device: str,
        **kwargs
    ):
        self.batch_size = batch_size
        self.device = device
        self.name = model_path.name

        super(Main, self).__init__(
            model_path=model_path,
            batch_size=batch_size,
            data_path=data_path,
            checkpoint=checkpoint,
            device=device,
            **kwargs,
        )

        # Load tokenizer, model and data (or last available checkpoint)
        # LoaderLLM.__init__(self, model_path, data_path, checkpoint, **kwargs)
        self.load_tokenizer(model_path)
        self.set_init_prompt()
        self.load_model(model_path, **kwargs)
        self.load_data(data_path, batch_size=batch_size)

        # Load LoRA parameters (if exists)
        # try:
        #     self.llm = PeftModel.from_pretrained(self.llm, model_path)
        #     self.llm = self.llm.merge_and_unload()
        #     self.logger.info("Previous trained LoRA weights are loaded and merged")

        # except Exception as e:
        #     print(e)
        #     self.logger.info("There is no previous trained LoRA weights")

        # Set LoRA parameters
        # lora_config = LoraConfig(
        #     r=8,  # r=8,
        #     lora_alpha=16,  # lora_alpha=16,
        #     lora_dropout=0.1,  # lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        # self.llm = get_peft_model(self.llm, lora_config)
        # self.logger.info("LoRA weights initiated")

        # self.print_trainable_parameters()
        # self.llm = self.llm.to(self.device)

    def run(self):
        if self.eval_data:
            self.logger.info("Eval test before training")
            self.eval()

        pass

    def eval(self):
        for eval_question in self.eval_data:
            print(eval_question)
            data = (self.init_prompt + eval_question)['assistant']
            # print(data)
            self.logger.info(f"{data}")
            inputs = torch.tensor([data.tokens])
            # print(inputs)
            # print(data.text)
            # print(data._tokens)
            print(f"_tokens size: {len(data._tokens)}")
            print(f"input size: {inputs.size(1)}")
            # print(data.mask)
            print(f"mask size: {len(data.mask)}")
            print(f"get n tokens: {data.get_n_tokens()}")

            generated_encoded = self.llm.generate(
                inputs=inputs,
                max_length=data.get_n_tokens() + 32,
            )[0]  # [0, data.get_n_tokens():]
            # print(generated_encoded)

            ans = self.tokenizer.decode(generated_encoded)

            self.logger.info(f"{ans}\n")
            # break

    def load_data(self, path, batch_size: int = 1):
        self.data = DataSet.from_jsonl(path, batch_size=batch_size)  #, tokenizer=self.tokenizer)
        self.logger.debug(f"Data loaded from {path}")

        # Set eval data
        self.eval_data = EVAL_DATASET

        # TODO: shuffle data (per batch ?)
        pass

    def load_tokenizer(self, path):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

        # /!\ LLaMa model have not pad token
        if self.tokenizer.pad_token is None:
            self.logger.info(f"Set pad with eos token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.debug(f"Tokenizer loaded from {path}")

    def set_init_prompt(self):
        self.init_prompt = Chat.from_jsonl(
            PROMPT_PATH / "short_prompt.jsonl",
            tokenizer=self.tokenizer,
        )

    def load_model(self, path, **kwargs):
        self.llm = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        self.logger.debug(f"<Model loaded from {path}>")



if __name__ == "__main__":
    from config import LOG, ROOT
    import logging.config

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

    # Get training arguments
    parser = TrainingParser(file=__file__)
    args = parser(data_path=ROOT / "data/instruct/full_data.jsonl")
    LOG.info(f"{parser}\n")

    if args.checkpoint:
        checkpoint = Checkpoint(
            path=args.checkpoint_path,
            timestep=args.checkpoint_timestep,
        )

    else:
        checkpoint = False

    main = Main(args.model, args.batch_size, args.data_path, checkpoint,
                args.device)
    main.run()
