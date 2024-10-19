#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-12-08 11:31:56
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-17 16:30:54

""" Trainer object. """

# Built-in packages
from typing import Self

# Third party packages
from tqdm import tqdm
import torch
from torch import Tensor
# from torch.optim import AdamW

# Local packages
from config import LOG_NO_CONSOLE
from save_load import Checkpoint
from utils import set_mask

__all__ = []


class Losses:
    """ Loss history.

    Parameters
    ----------
    current_loss : float, optional
        Current loss (the last one added).
    loss_history : list of float, optional
        History of losses.

    Methods
    -------
    __str__
    __iadd__
    append

    Attributes
    ----------
    current_loss : float
        Current loss (the last one added).
    loss_history : list of float
        History of losses.

    """

    def __init__(
        self: Self,
        current_loss: float = None,
        loss_history: list[float] = None
    ):
        self.current_loss = current_loss
        self.loss_history = [] if loss_history is None else loss_history

    def __str__(self) -> str:
        if self.current_loss is None:

            return "Current loss = None"

        return f"Current loss = {self.current_loss:.2e}"

    def append(self, loss: float) -> Self:
        """ Append a new loss to the history and the new loss is current one.

        Parameters
        ----------
        loss : float
            Loss to add to the history and become the current loss.

        Returns
        -------
        Losses
            Self object.

        """
        self.loss_history.append(loss)
        self.current_loss = loss

        return self

    def __iadd__(self, loss: float) -> Self:
        return self.append(loss)


class DataBrowser:
    """ Browse through the data.

    Parameters
    ----------
    dataset : list of str
        Data to browse.
    batch_size : int
        Size of the batch to browse data.

    Methods
    -------
    __iter__
    __next__
    set_description
    remaining_data

    Attrtibutes
    -----------
    dataset : list of str
        Data to browse.
    batch_size : int
        Size of the batch to browse data.
    start, end : int
        Index to respectivley start and end data browsing.
    i : int
        Index of the current data browsed.

    """

    def __init__(
        self,
        dataset: list[str],
        batch_size: int,
        start: int = 0,
        end: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        # TODO : - check validity of start and end (between 0 and T)
        #        - check start is lower than end
        self.start = start
        self.end = end if end is not None else len(self.dataset)

    def __iter__(self) -> Self:
        self.i = self.start
        self.pbar = tqdm(total=self.end - self.start)

        return self

    def __next__(self) -> list[str]:
        if self.i >= self.end:
            raise StopIteration

        i = self.i
        j = min(i + self.batch_size, self.end)

        self.i = j

        self.pbar.update(j - i)

        return self.dataset[i: j]

    def set_description(self, text: str):
        """ Wrtie description on pbar console and to logging.

        Parameters
        ----------
        text : str
            Text to write.

        """
        self.pbar.set_description(text)
        LOG_NO_CONSOLE.info(
            f"{self.i}/{len(self.dataset)} data - {text}"
        )

    def remaining_data(self) -> list[str]:
        """ Get data not yet used.

        Returns
        -------
        list of str
            Data not yet used.

        """
        return self.dataset[self.i:]


class Trainer:
    """ Train a LLM on a dataset.

    Parameters
    ----------
    llm : transformers.ModelForCausalLM
        Model to train.
    tokenizer : transformers.Tokenizer
        Object to tokenize text data.
    dataset : list of str
        List of data exemples (text).
    batch_size : int
        Number of data to train simultaneously.
    accumulation_steps : int, optional
        The number of mini-batches over which you accumulate gradients, default
        is 1.

    Methods
    -------
    __iter__
    __next__
    run
    set_mask
    set_optimizer
    training_step

    Attributes
    ----------
    accumulation_steps : int
        The number of mini-batches over which you accumulate gradients, default
        is 1.
    batch_size : int
        Number of data to train simultaneously.
    dataset : list of str
        List of data exemples (text).
    losses : Losses
        History loss object.
    llm : transformers.ModelForCausalLM
        Model to train.
    n_accumulated_grad : int
        Number of step accumulated gradient.
    optimizer : torch.optim.Optimizer
        Optimizer object.
    tokenizer : transformers.Tokenizer
        Object to tokenize text data.

    """

    losses = None
    n_accumulated_grad = None
    optimizer = None

    def __init__(
        self: Self,
        llm,
        tokenizer,
        dataset: list[str],
        batch_size: int,
        accumulation_steps: int = 1,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size

        self.accumulation_steps = accumulation_steps

    def __iter__(self) -> Self:
        self.losses = Losses()
        self.n_accumulated_grad = 0
        self._data_browser = DataBrowser(self.dataset, self.batch_size)
        self._data_browser.__iter__()

        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        data = self._data_browser.__next__()

        # Set input data batch
        encoded_data = self.tokenizer(data, return_tensors='pt', padding=True)
        input_ids = encoded_data['input_ids']
        attention_mask = self.set_mask(encoded_data.attention_mask, input_ids)

        # Display current loss and token size of data
        self._data_browser.set_description(
            f"{self.losses} - Token size = {encoded_data['input_ids'].size(1)}"
        )

        return input_ids, attention_mask

    def run(self, device: str, checkpoint: bool | Checkpoint):
        for input_ids, attention_mask in self:
            with torch.enable_grad():
                self.training_step(
                    input_ids.to(device),
                    attention_mask.to(device)
                )

            if checkpoint:
                data = self._data_browser.remaining_data()
                checkpoint(self.llm, data, tokenizer=self.tokenizer)

    def set_mask(self, attention_mask: Tensor, input_ids: Tensor) -> Tensor:
        """ Set mask attention randomly.

        Parameters
        ----------
        attention_mask : torch.Tensor
            Attention masks to update.
        input_ids : torch.Tensor
            Input IDs.

        Returns
        -------
        torch.Tensor
            Updated attention masks.

        """
        for i in range(attention_mask.size()[0]):
            attention_mask[i] = set_mask(attention_mask[i])

        return attention_mask

    def set_optimizer(self, optimizer, parameters=None, **kwargs):
        self.llm.train()

        if parameters is None:
            parameters = self.llm.parameters()

        self.optimizer = optimizer(parameters, **kwargs)

    def training_step(self, input_ids: Tensor, attention_mask: Tensor):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # Compute gradient and update weights
        loss = outputs.loss
        loss.backward()
        self.n_accumulated_grad += self.batch_size

        if self.n_accumulated_grad >= self.accumulation_steps:
            self.optimizer.step()

            # Update learning rate
            # lr_scheduler.step()

            # Reset gradient to zero
            self.optimizer.zero_grad()

            self.n_accumulated_grad = 0

        self.losses += loss.detach()


if __name__ == "__main__":
    pass
