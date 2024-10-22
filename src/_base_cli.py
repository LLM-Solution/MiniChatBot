#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-22 17:48:53
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-22 19:32:03

""" Description. """

# Built-in packages
from logging import getLogger
from pathlib import Path
from random import random
from time import sleep, strftime
from typing import Generator

# Third party packages
from llama_cpp import Llama

# Local packages

__all__ = []


class _BaseCommandLineInterface:
    """ Command line interface object to chat with the LLM.

    Parameters
    ----------
    model_path : Path or str
        Path to load weight of the model.
    lora_path : Path or str, optional
        Path to load LoRA weights.
    init_prompt : str, optional
        Initial prompt to feed the LLM.
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.
    n_ctx : int, optional
        Maximum number of input tokens for LLM.

    Methods
    -------
    __call__
    answer
    ask
    exit
    reset_conv
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

    similarity_search = None

    def __init__(
        self,
        model_path: Path | str,
        lora_path: Path | str = None,
        init_prompt: str = None,
        verbose: bool = False,
        n_ctx: int = 1024,
        n_threads=6,
        **kwargs,
    ):
        self.logger = getLogger('cli')
        self.verbose = verbose
        self.n_ctx = n_ctx
        self.logger.debug(f"verbose={verbose}, n_ctx={n_ctx}, "
                          f"n_threads={n_threads}, init_prompt={init_prompt}, "
                          f"kwargs={kwargs}")
        self.logger.debug(f"model_path={model_path}")
        self.logger.debug(f"lora_path={lora_path}")

        # Set LLM model
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            verbose=verbose,
            n_threads=n_threads,
            lora_path=lora_path,
            **kwargs,
        )

        self.today = strftime("%B %d, %Y")
        self.user_name = "User"
        self.ai_name = "MiniChatBot"
        self.logger.debug(f"user_name={self.user_name}&ai_name={self.ai_name}")
        self.init_prompt = init_prompt

        self.reset_conversation()

        self.stop = [f"{self.user_name}:", f"{self.ai_name}:"]

    def run(self):
        """ Start the command line interface for the AI chatbot. """
        self._display(f"\n\nWelcome, I am {self.ai_name} your custom AI, "
                      f"you can ask me anything about LLM Solution.\nPress ctrl + "
                      f"C or write 'exit' to exit\n\n")

        self.logger.debug("<Run>")

        try:
            while True:
                str_time = strftime("%H:%M:%S")
                question = input(f"{str_time} | {self.user_name}: ")

                if question == "exit":
                    self.exit(f"Goodbye {self.user_name} ! I hope to see you "
                              f"soon !\n")

                output = self.ask(question, stream=True)
                self.answer(output)

        except Exception as e:
            self.exit(f"\n\nAN ERROR OCCURS.\n\n{type(e)}: {e}")

    def answer(self, output: Generator[str, None, None]):
        """ Display the answer of the LLM.

        Parameters
        ----------
        output : generator
            Output of the LLM.

        """
        str_time = strftime("%H:%M:%S")
        self._display(f"{str_time} | {self.ai_name}:", end="", flush=True)
        answer = ""
        for text in output:
            answer += text
            self._display(text, end='', flush=True)

        self.conv_hist += f"\n{self.ai_name}: {answer}"

        self.logger.debug(f"{self.ai_name}: {answer}")

        self._display("\n")

    def ask(
        self,
        question: str,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """ Ask a question to the LLM.

        Parameters
        ----------
        question : str
            Question asked by the user to the LLM.
        stream : bool, optional
            If false (default) the full answer is waited before to be printed,
            otherwise the answer is streamed.

        Returns
        -------
        str or Generator of str
            The output of the LLM, if `stream` is `True` then return generator
            otherwise return a string.

        """
        self.logger.debug(f"ask : {self.user_name}: {question}")
        self.logger.debug(f"type {type(question)}")

        # Get prompt at the suitable format
        prompt = f"{self.user_name}: {question}"

        self.conv_hist += "\n" + prompt

        # Remove older conversation if exceed context limit
        self._check_conv_limit_context()

        # Feed the LLM with all the conversation historic available
        prompt = str(self.conv_hist)

        return self(str(prompt), stream=stream)

    def __call__(
        self,
        prompt: str,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """ Generate an answer by the LLM for a given raw prompt.

        Parameters
        ----------
        prompt : str
            Raw prompt to feed the LLM.
        stream : bool, optional
            If false (default) the full answer is waited before to be printed,
            otherwise the answer is streamed.

        Returns
        -------
        str or Generator of str
            The output of the LLM, if `stream` is `True` then return generator
            otherwise return a string.

        """
        self.logger.debug(f"prompt {prompt}")

        r = self.llm(prompt, stop=self.stop, stream=stream, max_tokens=None)

        if stream:
            return self._stream_call(r)

        else:

            return r['choices'][0]['text']

    def _stream_call(self, response: Generator) -> Generator[str, None, None]:
        for g in response:
            yield g['choices'][0]['text']

    def _check_conv_limit_context(self):
        # FIXME : How deal with too large conversation such that all the
        #         conversation is removed ?
        while self._get_n_token(str(self.conv_hist)) > self.n_ctx:
            poped_conv = self.conv_hist.pop_fifo()
            self.logger.debug(f"Pop the following part: {poped_conv}")

    def _get_n_token(self, sentence: str) -> int:
        return len(self.llm.tokenize(sentence.encode('utf-8')))

    def exit(self, txt: str = None):
        """ Exit the CLI of the chatbot.

        Parameters
        ----------
        txt : str, optional
            Text to display before exiting.

        """
        if txt is not None:
            self._display(f"{self.ai_name}: ")
            self._stream(txt)

        if self.verbose:
            print(f"\n\nThe conversation was:\n\n{self.conv_hist}\n\n")

        self.logger.debug("<Exit>")

        exit()

    def reset_conversation(self):
        """ Reset the current conversation with the `init_prompt`. """
        self.logger.debug("<Reset conversation>")

        self.conv_hist = self.init_prompt

    def _stream(self, txt: str):
        for chars in txt:
            self._display(chars)
            sleep(random() / 20)

    def _display(self, txt: str, end: str = "", flush: bool = True):
        print(txt, end=end, flush=flush)


if __name__ == "__main__":
    from config import MODEL_PATH, PROMPT

    _BaseCommandLineInterface()