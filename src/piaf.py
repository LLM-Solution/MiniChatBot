#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-11-30 10:29:12
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-18 14:06:11

""" Test training with LoRA method. """

# Built-in packages
from pathlib import Path
from random import seed, shuffle
from tqdm import tqdm

# Third party packages
from peft import get_peft_model, LoraConfig, PeftModel
import torch
from torch.optim import AdamW
from transformers import BitsAndBytesConfig

# Local packages
from config import LOG, TrainingParser, ACCUMULATION_STEPS, LR, DATA_PATH
from main import Main, PROMPT
from save_load import Checkpoint
from trainer import Trainer as BasisTrainer
from utils import find_sequence, generate, set_mask, shuffle_per_batch

__all__ = []


class TrainerQA(BasisTrainer):
    def __init__(self, *args, **kwargs):
        super(TrainerQA, self).__init__(*args, **kwargs)
        self.begin_of_answer = self.tokenizer(
            "ok[/Q][A]",
            return_tensors='pt',
        ).input_ids[0, -3:]
        self.end_of_answer = self.tokenizer(
            "ok [/A]",
            return_tensors='pt',
        ).input_ids[0, -3:]

        print("\nTrainer QA is initiated\n")

    def set_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        # rate: float = 0.35,
        rate: float = 1.,
    ) -> torch.Tensor:
        """ Set mask attention randomly.

        Parameters
        ----------
        attention_mask : torch.Tensor
            Attention masks to update.
        input_ids : torch.Tensor
            Input IDs.
        rate : float
            Rate of tokens to mask.

        Returns
        -------
        torch.Tensor
            Updated attention masks.

        """
        for i in range(input_ids.size(0)):
            # Looking for the beginning of the answer part
            idx_boa = find_sequence(self.begin_of_answer, input_ids[i])

            # Fallback if end of answer tag is not finded
            if idx_boa is None:
                LOG.info(f"BEO token not found\n"
                         f"{self.tokenizer.decode(input_ids[i])}\n")
                idx_boa = 0

            # Looking for the end of the answer part
            # include the tag end of the answer
            idx_eoa = find_sequence(self.end_of_answer, input_ids[i],
                                    start=idx_boa)

            # Fallback if end of answer tag is not finded
            if idx_eoa is None:
                LOG.info(f"EOA token not found\n"
                         f"{self.tokenizer.decode(input_ids[i, idx_boa:])}\n")
                idx_eoa = attention_mask[i].sum()

            # Set mask
            attention_mask[i] = set_mask(
                attention_mask[i],
                rate=rate,
                beginning_idx=idx_boa,
                end_idx=idx_eoa,
            )

        return attention_mask

    def run(self, device: str, checkpoint: bool | Checkpoint):
        for input_ids, attention_mask in self:
            with torch.enable_grad():
                self.training_step(
                    input_ids.to(device),
                    attention_mask.to(device)
                )

            if checkpoint:
                data = self._data_browser.remaining_data()
                data = [unformater(arg) for arg in data]
                checkpoint(self.llm, data, tokenizer=self.tokenizer)


def get_answer(answers: list[str]):
    if len(answers) == 1:
        return answers[0]['text']

    ans = [answers[0]['text']]

    for a in answers[1:]:
        if a['text'] in ans:
            continue

        ans.append(a['text'])

    if len(ans) == 1:
        return ans[0]

    else:
        return f"{', '.join(ans[:-1])} et {ans[-1]}"


def get_question(question: str):
    if question[-1] != "?":
        question += ' ?'

    return question


def formater(question: str, context: str = PROMPT, answer: str = ''):
    if not answer:
        return f"[C]{context}[/C][Q]{question}[/Q][A]"

    return f"[C]{context}[/C][Q]{question}[/Q][A]{answer} [/A]"


def unformater(data: str) -> tuple[str, str, str]:
    context, rest = data[3:].split("[/C][Q]")
    question, answer = rest[:-5].split("[/Q][A]")

    return {'question': question, 'answer': answer}


def _parse_data(data: dict[str, str]):
    full_scripts = []

    for args in tqdm(data['data']):
        for p in args['paragraphs']:
            context = p['context']
            for qas in p['qas']:
                question = get_question(qas['question'])
                answer = get_answer(qas['answers'])

                # full_scripts.append(formater(context, question, answer))
                full_scripts.append({
                    "question": question,
                    "context": context,
                    "answer": answer,
                })

    return full_scripts


def parse_data(data: list[dict[str, str]]):
    return [{"question": d["User"], "answer": d["MiniChatBot"]} for d in data]


class Main(Main):

    def __init__(self, model_name: Path, *args, **kwargs):
        super(Main, self).__init__(model_name, *args, **kwargs)

        try:
            self.llm = PeftModel.from_pretrained(self.llm, model_name)
            self.llm = self.llm.merge_and_unload()
            LOG.info("Previous trained LoRA weights are loaded and merged")

        except Exception as e:
            print(e)
            LOG.info("There is no previous trained LoRA weights")

        # Set LoRA parameters
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        LOG.info("LoRA weights initiated")

        self.print_trainable_parameters()
        self.llm = self.llm.to(self.device)

    def __call__(self, eval_sentences: list = None):
        # Test eval model before training
        if eval_sentences is not None:
            LOG.info("Eval test before training")
            self.eval(*eval_sentences)

        # Training
        try:
            trainer = TrainerQA(
                self.llm,
                self.tokenizer,
                self.data,
                self.batch_size,
                accumulation_steps=ACCUMULATION_STEPS,
            )

            trainer.set_optimizer(AdamW, self.llm.parameters(), lr=LR)

            trainer.run(device=self.device, checkpoint=self.checkpoint)

            # Save trained model
            path = (Path("./models/MiniChatBot-1B"))
            self.checkpoint.save_trained_model(self.llm, path, self.tokenizer)

        except AttributeError as e:
            LOG.error(f"The following error occurs: {type(e)} - {e}")

        # Test eval model after training
        if eval_sentences is not None:
            LOG.info("Eval test after training")
            self.eval(*eval_sentences)

    def eval(self, *data: dict[str], max_length: int = 64):
        """ Evaluate LLM by generate some sentences.

        Parameters
        ----------
        *data : dict of str
            Data with `question`, `context` and `answer` to evaluate the LLM.
        max_length : int, optional
            Maximum number of tokens generated by the LLM, default is 64
            tokens.

        """
        for args in data:
            text = formater(args['question'], PROMPT)
            length = len(self.tokenizer(text).input_ids)

            ans = generate(
                self.llm,
                self.tokenizer,
                text,
                max_length=length + max_length,
                device=self.device,
            )
            LOG.info(f"\n{ans}\n\n- Expected answer: {args['answer']}\n")

    def process_data(self):
        """ Process and shuffle data. """
        # Parse data
        seed(42)
        self.data = parse_data(self.data)
        shuffle(self.data)

        self.eval_data = self.data[:5]
        self.data = self.data[5:]
        # self.data = self.data[:100]
        self.data = [formater(**kwargs) for kwargs in self.data]
        self.data = shuffle_per_batch(
            self.data,
            self.tokenizer,
            batch_size=self.batch_size
        )


if __name__ == "__main__":
    import logging.config
    import yaml

    # Load logging configuration
    with open('./logging.ini', 'r') as f:
        log_config = yaml.safe_load(f.read())

    logging.config.dictConfig(log_config)

    # Get training arguments
    parser = TrainingParser(file=__file__)
    args = parser(data_path=DATA_PATH)
    LOG.info(f"{parser}\n")

    if args.checkpoint:
        checkpoint = Checkpoint(
            path=args.checkpoint_path,
            timestep=args.checkpoint_timestep,
        )

    else:
        checkpoint = False

    if args.device != "cpu":
        # kw_gpu = dict(
        #     load_in_4bit=True,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        # )

        kw_gpu = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }

    else:
        kw_gpu = {}

    main = Main(args.model, args.batch_size, args.data_path, checkpoint,
                args.device, **kw_gpu)
    main(main.eval_data)
