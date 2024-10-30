#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-11-30 10:29:12
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-30 15:55:38

""" Training MiniChatBot with LoRA method. """

# Built-in packages
from pathlib import Path
from random import seed, shuffle
from tqdm import tqdm

# Third party packages
from peft import get_peft_model, LoraConfig, PeftModel
from pyllmsol import Trainer as BasisTrainer
from pyllmsol.training.checkpoint import Checkpoint, LoaderLLM
from pyllmsol.training.utils import (find_token, find_sequence, generate,
                                     set_mask, shuffle_per_batch)
import torch
from torch.optim import AdamW
from transformers import BitsAndBytesConfig

# Local packages
from config import LOG, TrainingParser, ACCUMULATION_STEPS, LR, DATA_PATH, PROMPT
# from main import Main
# from save_load import Checkpoint

__all__ = []


EVAL_DATA = [
    {
        'User': "Who are you and what is LLM Solutions?",
        'MiniChatBot': "",
    },
    {
        'User': 'Where is the best restaurent in the town?',
        'MiniChatBot': '',
    },
    {
        'User': 'Comment ça va ?',
        'MiniChatBot': '',
    },
    {
        'User': "J'ai un e-commerce en ligne, à quoi peut m'aider un chatbot ?",
        "MiniChatBot": "",
    },
    {
        'User': 'What do you know about Arthur Bernard?',
        'MiniChatBot': '',
    },
]

# begin_context = "[C]"
# end_context = "[/C]"
# begin_question = "[Q]"
# end_question = "[/Q]"
# begin_answer = "[A]"
# end_answer = "[/A]"
begin_context = ""
end_context = "\n"
begin_question = "User: "
end_question = "\n"
begin_answer = "MiniChatBot: "
end_answer = "\n"

# new_tokens = [begin_context, end_context, begin_question, end_question,
#               begin_answer, end_answer]
new_tokens = []


class TrainerQA(BasisTrainer):
    def __init__(self, *args, **kwargs):
        super(TrainerQA, self).__init__(*args, **kwargs)
        self.begin_of_question = self.tokenizer(
            f"{begin_question}",
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids[0, :-1]
        self.end_of_question = self.tokenizer(
            f"{end_question}",
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids[0, :-1]
        self.begin_of_answer = self.tokenizer(
            f"{begin_answer}",
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids[0, :-1]
        self.end_of_answer = self.tokenizer(
            f"{end_answer}",
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids[0, :-1]

        LOG.info("Trainer QA is initiated\n")

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
            # idx_boa = find_token(self.begin_of_answer, input_ids[i]) + 1
            idx_boa = find_sequence(self.begin_of_answer, input_ids[i])
            # idx_boa = find_sequence(self.begin_of_question, input_ids[i])

            # Fallback if end of answer tag is not finded
            if idx_boa is None:
                LOG.info(f"BOA token not found\n"
                         f"{self.tokenizer.decode(input_ids[i])}\n")
                idx_boa = 0

            # Looking for the end of the answer part
            # include the tag end of the answer
            # idx_eoa = find_token(self.end_of_answer, input_ids[i],
            #                      start=idx_boa) + 1
            idx_eoa = find_sequence(self.end_of_answer, input_ids[i],
                                 start=idx_boa)

            # Fallback if end of answer tag is not finded
            if idx_eoa is None:
                LOG.info(f"EOA token not found\n"
                         f"{self.tokenizer.decode(input_ids[i, idx_boa:])}\n")
                idx_eoa = attention_mask[i].sum()

            # Set mask
            if rate >= 1.:
                attention_mask[i, idx_boa: idx_eoa] = 0

            else:
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
    txt = ""

    if context:
        txt += f"{begin_context}{context}{end_context}"

    txt += f"{begin_question}{question}{end_question}{begin_answer}"

    if answer:
        txt += f"{answer}{end_answer}"

    return txt


def unformater(data: str) -> tuple[str, str, str]:
    # context, rest = data[3:].split(f"{end_context}{begin_question}")
    # question, answer = rest[:-5].split(f"{end_question}{begin_answer}")

    # return {'question': question, 'answer': answer}
    context = data.split(f"{end_context}{begin_question}")[0]
    split_data = data.split(f"{end_question}{begin_answer}")
    answer = split_data[-1]
    question = split_data[-2].split(f"{end_answer}{begin_question}")[-1]

    return {'User': question, 'MiniChatBot': answer}


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


class Main(LoaderLLM):
    """ Class to train a chatbot to answer.

    Parameters
    ----------
    model_path : Path
        Path of the model to load.
    batch_size : int
        Size of batch.
    data_path : Path
        Path of the dataset to load.
    checkpoint : bool or Checkpoint
        If True or Checkpoint object then make checkpoint of trained model and
        data at regular timestep.
    device : str
        Device to make computation (cpu or gpu).
    **kwargs
        Keyword arguments for the class method
        `transformers.AutoModelForCausalLM.from_pretrained`, cf transformers
        documentation.

    """

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
        self.checkpoint = checkpoint
        self.device = device
        self.name = model_path.name

        # Load tokenizer, model and data (or last available checkpoint)
        LoaderLLM.__init__(self, model_path, data_path, checkpoint, **kwargs)

        # Process data
        self.process_data()

        # Load LoRA parameters (if exists)
        try:
            self.llm = PeftModel.from_pretrained(self.llm, model_path)
            self.llm = self.llm.merge_and_unload()
            LOG.info("Previous trained LoRA weights are loaded and merged")

        except Exception as e:
            print(e)
            LOG.info("There is no previous trained LoRA weights")

        # Set LoRA parameters
        lora_config = LoraConfig(
            r=128,  # r=8,
            lora_alpha=16,  # lora_alpha=16,
            lora_dropout=0.1,  # lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        LOG.info("LoRA weights initiated")

        self.print_trainable_parameters()
        self.llm = self.llm.to(self.device)

        # Add tokens
        if new_tokens:
            n_new_tokens = self.tokenizer.add_tokens(new_tokens)
            print(f"{n_new_tokens} new tokens added")
            self.llm.resize_token_embeddings(len(self.tokenizer))

    def __call__(self, eval_sentences: list = None):
        """ Run the training and evaluate the model before and after.

        Parameters
        ----------
        eval_sentences : list of str, optional
            List of text to evaluate model.

        """
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

    def eval(self, *data: dict[str], max_length: int = 32):
        """ Evaluate LLM by generate some sentences.

        Parameters
        ----------
        *data : dict of str
            Data with `question`, `context` and `answer` to evaluate the LLM.
        max_length : int, optional
            Maximum number of tokens generated by the LLM, default is 32
            tokens.

        """
        for args in data:
            text = formater(args['User'], PROMPT)
            length = len(self.tokenizer(text).input_ids)

            ans = generate(
                self.llm,
                self.tokenizer,
                text,
                max_length=length + max_length,
                device=self.device,
            )
            ans = ans[len(PROMPT):]

            LOG.info(f"\n{ans}\n\n- Expected answer: {args['MiniChatBot']}\n")

    def process_data(self):
        """ Process and shuffle data. """
        # Parse data
        seed(42)
        self.data = parse_data(self.data)
        shuffle(self.data)

        # self.eval_data = self.data[:5]
        # self.data = self.data[5:]
        self.eval_data = EVAL_DATA
        # self.data = self.data[:25]
        self.data = [formater(**kwargs) for kwargs in self.data]
        self.data = shuffle_per_batch(
            self.data,
            self.tokenizer,
            batch_size=self.batch_size
        )

    def print_trainable_parameters(self):
        """ Display trainable parameters. """
        trainable_params = 0
        all_param = 0

        for _, param in self.llm.named_parameters():
            all_param += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()

        LOG.info(f"\n\nTrainable params: {trainable_params:,} || All params: "
                 f"{all_param:,} || Trainable: "
                 f"{trainable_params / all_param:.2%}\n\n")


if __name__ == "__main__":
    from config import ROOT
    import logging.config

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

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
