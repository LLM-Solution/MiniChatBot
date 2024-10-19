#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-10-09 17:57:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-01-06 10:42:23

""" Objects to save and/or make a checkpoint of models. """

# Built-in packages
from json import loads, dumps
from logging import getLogger
from pathlib import Path
from time import time

# Third party packages
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local packages

__all__ = []


LOG = getLogger('train')


def loader(model_path, data_path, checkpoint=False, **kwargs):
    try:
        llm, data = checkpoint.load(**kwargs)

    except Exception:
        llm = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        data = loads(Path(data_path).open("r").read())
        LOG.debug(f"<Init model and full dataset ({len(data):,}) loaded>")

    return llm, data


class Checkpoint:
    """ Object to create checkpoint at regular timestep.

    Parameters
    ----------
    path : str or Path
        Path of the folder to save checkpoint of model and data, default is
        `checkpoints` folder at the root of the project.
    timestep : int
        Timestep in seconds to save the checkpoint, default is 300 (5 minutes).

    Methods
    -------
    __bool__
    __call__
    delete
    load
    save
    save_trained_model

    Attributes
    ----------
    path : Path
        Path of the folder to save checkpoint of model and data.
    timestep : int
        Timestep in seconds to save the checkpoint.
    ts : int
        Timestamp of the last checkpoint.

    """

    def __init__(self, path: str | Path = "./checkpoint/",
                 timestep: int = 300):
        # Set variables
        self.path = Path(path) if isinstance(path, str) else path
        self.timestep = timestep
        self.ts = time()

        # Create path if not already exist
        self.path.mkdir(parents=True, exist_ok=True)
        LOG.debug("<Checkpoint object is initiated>")

    def __bool__(self):
        """ Check if the last checkpoint is older than the timestep.

        Returns
        -------
        bool
            `True` if the last checkpoint is older than the timestep, otherwise
            `False`.

        """
        return time() - self.ts > self.timestep

    def __call__(self, llm, data, tokenizer=None):
        """ Save checkpoint if the last checkpoint is older than the timestep.

        Parameters
        ----------
        llm : AutoModelForCausalLM
            Model to make the checkpoint.
        data :
            Data to make the checkpoint.
        tokenizer : transformers.Tokenizer
            Object to tokenize text data.

        """
        if self:
            self.save(llm, data, tokenizer=tokenizer)

        else:
            pass

    def save(self, llm, data, tokenizer=None):
        """ Save the checkpoint of the LLM model and data.

        Parameters
        ----------
        llm : AutoModelForCausalLM
            Model to make the checkpoint.
        data :
            Data to make the checkpoint.
        tokenizer : transformers.Tokenizer
            Object to tokenize text data.

        """
        LOG.debug(f"<Checkpoint is saving model and data>")

        # Save model
        llm.save_pretrained(self.path / "model")

        # save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(self.path / "model")

        # Save data
        with (self.path / "data.json").open("w") as f:
            f.write(dumps(data))

        LOG.debug(f"<Checkpoint saved at: '{self.path}'>")

        # update timestamp
        self.ts = time()

    def load(self, **kwargs):
        """ Load the checkpoint of LLM model and data.

        Parameters
        ----------
        **kwargs
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`
            method, cf transformer documentation.

        Returns
        -------
        AutoModelForCausalLM
            Model from the checkpoint.
        Data
            Data from the checkpoint.

        """
        # Load model
        llm = AutoModelForCausalLM.from_pretrained(
            self.path / "model",
            **kwargs
        )

        # Load LoRA weights
        # try:
        #     llm = PeftModel.from_pretrained(llm, self.path / "model")
        #     llm = llm.merge_and_unload()
        #     LOG.info("Previous trained LoRA weights are loaded and merged")

        # except Exception as e:
        #     print(e)
        #     LOG.info("There is no previous trained LoRA weights")

        # Load data
        with (self.path / "data.json").open("r") as f:
            data = loads(f.read())

        LOG.info(f"<Model and data ({len(data):,}) loaded from checkpoint>")

        return llm, data

    def delete(self):
        """ Delete irreversibly the checkpoint. """
        # TODO : python3.12 => use walk method
        if self.path.exists():
            if (self.path / "model").exists():
                # Delete model files
                for f in (self.path / "model").rglob("*"):
                    f.unlink()

                (self.path / "model").rmdir()

            # Delete data file
            if (self.path / "data.json").exists():
                (self.path / "data.json").unlink()

            # Delete folder
            self.path.rmdir()

    def save_trained_model(self, llm, path, tokenizer=None):
        """ Save the trained model and delete checkpoint.

        Must be called when training has finished.

        Parameters
        ----------
        llm : AutoModelForCausalLM
            Trained model to save.
        path : Path
            Path to save the trained model.
        tokenizer : transformers.Tokenizer
            Object to tokenize text data.

        """
        # Save model and tokenizer
        path.mkdir(parents=True, exist_ok=True)

        llm.save_pretrained(path)

        if tokenizer is not None:
            tokenizer.save_pretrained(path)

        # Delete checkpoint
        self.delete()

        LOG.info("<Trained model saved and checkpoint deleted>")


class LoaderLLM:
    """ Load tokenizer and model. """

    def __init__(
        self,
        model_name: Path,
        data_path: Path,
        checkpoint: bool | Checkpoint,
        **kw_load_model,
    ):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )

        # /!\ LLaMa model have not pad token
        if self.tokenizer.pad_token is None:
            LOG.info(f"Set pad with eos token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model and data (or last available checkpoint)
        self.llm, self.data = loader(model_name, data_path,
                                     checkpoint=checkpoint, **kw_load_model)


if __name__ == "__main__":
    pass
