#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-13 16:23:53
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-15 12:32:18

""" Description. """

# Built-in packages
from pathlib import Path
from random import seed, shuffle

# Third party packages
from peft import get_peft_model, LoraConfig, PeftModel
from pyllmsol._base import _Base
from pyllmsol.data.chat import Chat, DataSet
from pyllmsol.training.checkpoint import Checkpoint
from pyllmsol.training.instruct_trainer import TrainerInstruct
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local packages
from config import (TrainingParser, ACCUMULATION_STEPS, LR, DATA_PATH,
                    PROMPT_PATH, SAVE_MODEL_PATH)

__all__ = []


EVAL_MESSAGE = [
    [dict(role="user", content="Who are you and what's LLM Solution?")],
    [dict(role="user", content="Where is the best restaurent in town?")],
    [dict(role="user", content="Comment ça va ?")],
    [dict(role="user", content=("J'ai un e-commerce en ligne, à quoi peut "
                                "m'aider un chatbot dans mon business ?"))],
    [dict(role="user", content="What do you know about Arthur Bernard?")],
]


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
                checkpoint(self.llm, data.to_json(), tokenizer=self.tokenizer)


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
        self.checkpoint = checkpoint
        # TODO : load checkpoint

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
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.logger.info("LoRA weights initiated")

        self.print_trainable_parameters()
        self.llm = self.llm.to(self.device)

    def run(self):
        if self.eval_data:
            self.logger.info("Eval test before training")
            self.eval()

        # Training
        try:
            trainer = ChatTrainer(
                self.llm,
                self.tokenizer,
                self.data,
                self.batch_size,
                accumulation_steps=ACCUMULATION_STEPS,
            )

            trainer.set_optimizer(AdamW, self.llm.parameters(), lr=LR)

            trainer.run(device=self.device, checkpoint=self.checkpoint)

            # Save trained model
            # path = Path("./models/MiniChatBot-1B")
            path = SAVE_MODEL_PATH
            self.checkpoint.save_trained_model(self.llm, path, self.tokenizer)

        except AttributeError as e:
            self.logger.error(f"The following error occurs: {type(e)} - {e}")

            raise e

        if self.eval_data:
            self.logger.info("Eval test after training")
            self.eval()

    def eval(self):
        for eval_data in self.eval_data:
            eval_question = eval_data.items.pop()
            data = (self.init_prompt + eval_question)['assistant']
            self.logger.info(f"Question: {eval_question.items[-1].content}")
            inputs = torch.tensor([data.tokens])

            generated_encoded = self.llm.generate(
                inputs=inputs,
                max_length=data.get_n_tokens() + 32,
            )[0, data.get_n_tokens():]

            ans = self.tokenizer.decode(generated_encoded)

            self.logger.info(f"Generated: {ans}")

    def load_data(self, path, batch_size: int = 1):
        data = DataSet.from_jsonl(
            path,
            batch_size=batch_size,
            tokenizer=self.tokenizer,
            # end=10,  # TODO : remove
        )
        self.logger.debug(f"Data loaded from {path}")

        # Shuffle data
        seed(42)
        shuffle(data.items)

        self.logger.debug(f"Sample chat: {data[0]}")
        self.data = DataSet(
            [self.init_prompt + item for item in data.items],
            batch_size=batch_size,
            tokenizer=self.tokenizer,
            # end=50,  # TODO : to remove
        )

        self.logger.debug(f"Sample chat: {self.data[0]}")
        self.logger.debug(f"Init prompt added to data")

        # Set eval data
        self.eval_data = DataSet(EVAL_MESSAGE, tokenizer=self.tokenizer)

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

    def print_trainable_parameters(self):
        """ Display trainable parameters. """
        trainable_params = 0
        all_param = 0

        for _, param in self.llm.named_parameters():
            all_param += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()

        self.logger.info(f"\n\nTrainable params: {trainable_params:,} || All "
                         f"params: {all_param:,} || Trainable: "
                         f"{trainable_params / all_param:.2%}\n\n")


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
