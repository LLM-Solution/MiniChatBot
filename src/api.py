#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-23 16:25:55
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-23 18:10:58

""" Flask API object for MiniChatBot. """

# Built-in packages

# Third party packages
from flask import Flask, request, Response

# Local packages
from _base_api import API, cors_required
from _base_cli import _BaseCommandLineInterface
from config import LOG, GGUF_MODEL, PROMPT

__all__ = []


class MiniChatBotAPI(API, _BaseCommandLineInterface):
    def __init__(
        self,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads=6,
        debug: bool = False,
        **kwargs,
    ):
        self.debug = debug

        # Set CLI part
        _BaseCommandLineInterface.__init__(
            self,
            model_path=GGUF_MODEL,
            init_prompt=PROMPT,
            verbose=verbose,
            n_ctx=32768,
            n_threads=6,
            **kwargs,
        )

        # Set API part
        LOG.debug("Start init Flask API object")
        self.app = Flask(__name__)
        # self.add_route()
        self.add_post_cli_route()

        LOG.debug("Flask API object is initiated")

    def add_post_cli_route(self):
        """ Add POST routes to communicate with the CLI. """
        @self.app.route("/ask", methods=['POST', 'OPTIONS'])
        @cors_required
        def ask():
            """ Ask a question to the LLM.

            Examples
            --------
            >>> output = requests.post(
            ...     "http://0.0.0.0:5000/ask",
            ...     json={
            ...         "question": "Who is the president of USA ?",
            ...         "stream": True,
            ...     },
            ... )
            >>> for txt in output.iter_content():
            ...     print(txt.decode('utf8'), end='', flush=True)
            Robot: Joe Biden is the president of USA.

            """
            question = request.json.get("question")
            stream = request.json.get("stream", True)
            session_id = request.json.get("session_id")
            LOG.debug(f"ask: {question}")

            # FIXME : should be escaped ? to avoid code injection
            # return self.cli.ask(escape(question), stream=stream)

            output = self.ask(question, stream=stream)

            return self.answer(output, stream=stream)

    def answer(self, output, stream: bool = False):
        if stream:
            def generator():
                for chunk in output:
                    self.prompt_hist += chunk

                    yield chunk

            response = Response(generator(), content_type='text/event-stream')

        else:
            self.prompt_hist += output
            reponse = Response(output)

        return response


if __name__ == "__main__":
    import logging.config
    import yaml

    # Load logging configuration
    with open('./logging.ini', 'r') as f:
        log_config = yaml.safe_load(f.read())

    logging.config.dictConfig(log_config)

    debug = True

    with MiniChatBotAPI(debug=debug) as app:
        app.run(host='0.0.0.0', port=5000, debug=debug)
