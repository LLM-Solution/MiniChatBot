#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-18 17:26:54
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-30 08:20:32

""" Flask API object. """

# Built-in packages
from functools import wraps
from logging import getLogger
from pathlib import Path

# Third party packages
from flask import Flask, request, make_response, jsonify, stream_with_context, Response

# Local packages
from _base_cli import _BaseCommandLineInterface
from config import GGUF_MODEL, PROMPT

__all__ = []


LOG = getLogger('app')


# CORS decorator function
def cors_required(f):
    @wraps(f)
    def wrapped_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            # Preflight request
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods",
                                 "POST, OPTIONS, GET")

            return response

        # Actual request
        response = make_response(f(*args, **kwargs))
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response

    return wrapped_function


class API:
    """ Flask API object to run a LLM chatbot.

    Parameters
    ----------
    root : Path, optional
        Path of the root to loads LLM weights, default is './'.
    model_path : str, optional
        Path to the model to load (must be GGUF format). Default is a Mistral
        7B model.
    lora_path : str, optional
        Path to LoRA weights to load.
    n_ctx : int, optional
        Max number of tokens in the prompt, default is 4096.
    add_context : bool, optional
        If True then add context from embedded docs. Default is False.

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

    """

    lock_file = Path("/tmp/api.py.lock")

    def __init__(
        self,
        root: Path = Path('.'),
        model_path: str | Path = GGUF_MODEL,
        lora_path: str | Path = None,  # "models/ggml-adapter-model.bin",
        init_prompt: str = PROMPT,
        n_ctx: int = 32768,
        debug: bool = False,
    ):
        self.init_prompt = init_prompt
        self.debug = debug

        LOG.debug("Start init Flask API object")
        self.app = Flask(__name__)
        self.add_route()

        # Set CLI object
        lora_path = str(root / lora_path) if lora_path else None
        self.cli = _BaseCommandLineInterface(
            root / model_path,
            lora_path=lora_path,
            init_prompt=self.init_prompt,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            n_threads=None,
        )

        # Add GET and POST to CLI routes
        self.add_get_cli_route()
        self.add_post_cli_route()

        LOG.debug("Flask API object is initiated")

    def add_route(self):
        """ Add classical routes. """
        @self.app.route("/shutdown", methods=["POST"])
        def shutdown():
            """ Shutdown flask API server. """
            LOG.debug("Shutdown call")
            func = request.environ.get("werkzeug.server.shutdown")

            if func is None:

                raise RuntimeError("Not running with the Werkzeug Server")

            func()

            return "Server shutting down..."

        @self.app.route("/health", methods=['GET'])
        def health_check():
            """ Check status. """
            LOG.debug("GET health")

            return Response(status=200)

        @self.app.route("/ping", methods=['GET'])
        def ping():
            """ Ping the server. """
            LOG.debug("pong")

            return 'pong'

        @self.app.route("/", methods=['GET'])
        def index():
            """ Page index. """
            LOG.debug("GET index page")

            with open("./index.html", "r") as f:
                text = f.read()

            return text

    def add_get_cli_route(self):
        """ Add GET routes to communicate with the CLI. """
        @self.app.route("/reset_prompt", methods=['GET'])
        def reset_prompt():
            """ Reset the prompt of the LLM.

            Examples
            --------
            >>> requests.get("http://0.0.0.0:5000/reset_prompt")
            <Response [200]>

            """
            self.cli.reset_prompt()
            LOG.debug(f"GET reset prompt")

            return Response(status=200)

        @self.app.route("/get_prompt", methods=['GET'])
        def get_prompt():
            """ Return the prompt of the LLM.

            Examples
            --------
            >>> requests.get("http://0.0.0.0:5000/get_prompt")
            [Q]Who is the president of USA ?[/Q][A]Joe Biden.[/A]

            """
            prompt = self.cli.prompt_hist.to_json()
            LOG.debug(f"GET prompt : {prompt}")

            return prompt

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

    def run(self, timer=0, **kwargs):
        """ Start to run the flask API. """
        if timer > 0:
            LOG.debug(f"Flask API is running for {timer} seconds")
            flask_thread = Thread(
                target=self.app.run,
                kwargs=kwargs,
                daemon=True,
            )
            flask_thread.start()

            timer_thread = Thread(
                target=self._timer_to_shutdown,
                args=(timer,),
            )
            timer_thread.start()
            timer_thread.join()

        else:
            LOG.debug("Flask API is running")
            self.app.run(**kwargs)

    def _timer_to_shutdown(self, duration):
        LOG.debug(f"API will shutdown in {duration}")
        sleep(duration)
        LOG.debug("End of timer, API server shutting down")
        exit(0)

    def __enter__(self):
        LOG.debug("Enter in control manager")
        self.lock_file.touch(exist_ok=self.debug)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lock_file.unlink(missing_ok=self.debug)
        LOG.debug("Exit of control manager")

        return False

if __name__ == '__main__':
    import logging.config
    import yaml

    # Load logging configuration
    with open('./logging.ini', 'r') as f:
        log_config = yaml.safe_load(f.read())

    logging.config.dictConfig(log_config)

    debug = True

    with API(debug=debug) as app:
        app.run(host='0.0.0.0', port=5000, debug=debug)
