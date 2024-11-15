#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-18 17:26:54
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-15 12:24:18

""" Flask API object. """

# Built-in packages
from functools import wraps
from pathlib import Path

# Third party packages
from flask import request, make_response, Response
from pyllmsol.inference._base_api import API as _BaseAPI

# Local packages
from cli import _BaseCommandLineInterface
from config import GGUF_MODEL

__all__ = []


# CORS decorator function
def cors_required(f):
    @wraps(f)
    def wrapped_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            # Preflight request
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type",
            )
            response.headers.add(
                "Access-Control-Allow-Methods",
                "POST, OPTIONS, GET",
            )

            return response

        # Actual request
        response = make_response(f(*args, **kwargs))
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response

    return wrapped_function


class API(_BaseAPI):
    """ Flask API object to run a LLM chatbot.

    Parameters
    ----------
    lora_path : str, optional
        Path to LoRA weights to load.
    n_ctx : int, optional
        Max number of tokens in the prompt, default is 32768.
    debug : bool, optional
        Debug mode for flask API, default is False.

    Methods
    -------
    add_get_cli_route
    add_post_cli_route
    add_route
    run

    Attributes
    ----------
    app : Flask
       Flask application.
    cli : CommandLineInterface
       Chatbot object.
    debug : bool, optional
        If True then don't block app if lock_file already exist. It means
        several servers can run simultanously.
    init_prompt : str
        Initial prompt to feed the LLM in the CLI.

    Notes
    -----
    API Routes:

    - **POST** `/shutdown`
        - Description: Shuts down the Flask API server.
        - Response: Returns the message "Server shutting down...".

    - **GET** `/health`
        - Description: Checks the health/status of the server.
        - Response: Returns HTTP status code 200.

    - **GET** `/ping`
        - Description: Pings the server to confirm it is running.
        - Response: Returns the string "pong".

    - **GET** `/reset_prompt`
        - Description: Resets the prompt of the LLM.
        - Response: HTTP status code 200.

    - **GET** `/get_prompt`
        - Description: Returns the current prompt of the LLM.
        - Response: The prompt string in JSON format.

    - **POST** `/set_init_prompt`
        - Description: Sets a new initial prompt for the LLM.
        - Body: `{"init_prompt": "<new_initial_prompt>"}`
        - Response: HTTP status code 200.

    - **POST** `/ask`
        - Description: Sends a question to the LLM.
        - Body: `{"question": "<question_text>", "stream": true/false,
            "session_id": "<session_id>"}`
        - Response: The LLM's answer, streamed or as a full response.

    - **POST** `/call`
        - Description: Sends a raw prompt to the LLM.
        - Body: `{"prompt": "<prompt_text>", "stream": true/false}`
        - Response: The LLM's answer, streamed or as a full response.

    """

    lock_file = Path("/tmp/api.py.lock")

    def __init__(
        self,
        lora_path: str | Path = None,
        n_ctx: int = 32768,
        debug: bool = False,
    ):
        super(API, self).__init__(
            model_path=GGUF_MODEL,
            init_prompt=PROMPT,
            lora_path=lora_path,
            n_ctx=n_ctx,
            debug=debug,
        )

    def add_post_cli_route(self):
        """ Add POST routes to communicate with the CLI. """
        @self.app.route("/set_init_prompt", methods=['POST'])
        def set_init_prompt():
            """ Set the prompt to the LLM.

            Examples
            --------
            >>> output = resquests.post(
            ...     "http://0.0.0.0:5000/set_init_prompt",
            ...     json={
            ...         "init_prompt": ("Conversation between an helpfull AI" 
            ...                         "assistant and a human.")
            ...     },
            ... )

            """
            init_prompt = request.json.get("init_prompt")
            LOG.debug(f"POST set prompt : {init_prompt}")
            self.cli.init_prompt = init_prompt

            return Response(status=200)

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

            answer = self.cli.ask(question, stream=stream)  # session_id

            if stream:
                return Response(answer, content_type='text/event-stream')

            else:
                return answer

        @self.app.route("/call", methods=['POST', 'OPTIONS'])
        @cors_required
        def call():
            """ Call the LLM with a given raw prompt.

            Examples
            --------
            >>> output = requests.post(
            ...     "http://0.0.0.0:5000/call",
            ...     json={
            ...         "prompt": "Who is the president of USA ?",
            ...         "stream": False,
            ...     },
            ... )
            >>> output.text
            Robot: Joe Biden is the president of USA.

            """
            prompt = request.json.get("prompt")
            stream = request.json.get("stream", True)
            LOG.debug(f"call: {prompt}")

            answer = self.cli(prompt, stream=stream)

            if stream:
                return Response(answer, content_type='text/event-stream')

            else:
                return answer


if __name__ == '__main__':
    import logging.config
    from config import ROOT

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

    debug = True

    with API(debug=debug) as app:
        app.run(host='0.0.0.0', port=5000, debug=debug)
