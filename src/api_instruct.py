#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-15 12:16:12
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-19 17:23:13

""" Flask API object for MiniChatBot. """

# Built-in packages
from datetime import datetime, timedelta
from logging import getLogger
from markupsafe import escape
from random import randint
import re
from secrets import token_hex

# Third party packages
from flask import Flask, request, Response

# Local packages
from _base_api import API, cors_required
from cli_instruct import InstructCLI as CommandLineInterface
from config import GGUF_MODEL
from utils import load_storage, save_storage, send_email_otp, save_message

__all__ = []


class MiniChatBotAPI(API, CommandLineInterface):
    """ MiniChatBot API object to chat with the retrained LLM.

    Parameters
    ----------
    lora_path : Path or str, optional
        Path to load LoRA weights.
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.
    n_ctx : int, optional
        Maximum number of input tokens for LLM, default is 32 768.
    n_threads : int, optional
        Number of threads to compute the inference.
    debug : bool, optional
        Debug mode for flask API.
    **kwargs
        Keyword arguments for llama_cpp.Llama object, cf documentation.

    Methods
    -------
    __call__
    answer
    ask
    exit
    load_storage
    reset_prompt
    run
    save_storage

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

    Notes
    -----
    API Routes:

    - **GET** `/health`
        - Description: Checks the health/status of the server.
        - Response: Returns HTTP status code 200.

    - **POST** `/ask`
        - Description: Allows the user to ask a question to the LLM.
        - Headers: `Authorization: Bearer <token>`
        - Body: `{"question": "Your question here", "stream": true/false,
            "session_id": "<session_id>"}`
        - Response: Returns the LLM's response, either streamed or as a full
            response.

    - **POST** `/send-otp`
        - Description: Sends an OTP to the provided email.
        - Body: `{"email": "user@example.com"}`
        - Response: Confirms that the OTP was sent or returns an error if the
            email format is invalid.

    - **POST** `/verify-otp`
        - Description: Verifies the OTP code provided by the user.
        - Body: `{"email": "user@example.com", "otp": "123456"}`
        - Response: Returns a success message with an authentication token if
            the OTP is correct, or an error if not.

    """

    def __init__(
        self,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads=4,
        debug: bool = False,
        **kwargs,
    ):
        self.debug = debug

        # Set CLI part
        CommandLineInterface.__init__(
            self,
            verbose=verbose,
            n_ctx=n_ctx,
            n_threads=n_threads,
            **kwargs,
        )

        # Storage of otp code and tokens
        self.load_storage()

        # Set API part
        self.logger.debug("Start init Flask API object")
        self.app = Flask(__name__)
        self.add_route()
        self.add_post_cli_route()

        self.logger.debug("Flask API object is initiated")

    def load_storage(self):
        """ Load and decrypt OTP code and tokens. """
        data = load_storage()
        self.otp_store = data["otp_store"]
        self.session_tokens = data["session_tokens"]

    def save_storage(self):
        """ Encrypt and save OTP codes and tokens. """
        save_storage({
            "otp_store": self.otp_store,
            "session_tokens": self.session_tokens,
        })

    def add_route(self):
        """ Add classical routes. """
        @self.app.route("/health", methods=['GET'])
        @cors_required
        def health_check():
            """ Check status.

            Returns
            -------
            flask.Response
                Status code 200.

            """
            self.logger.debug("GET health")

            return Response(status=200)

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
            auth_header = request.headers.get('Authorization')

            if not auth_header:
                self.logger.error(f"Missing Authorization header")

                return {'error': 'Unauthorized'}, 401

            token = auth_header.split(" ")[1]  # "Bearer <token>"
            email = request.json.get('email')

            if self.session_tokens.get(email) != token:
                self.logger.error(f"Wrong token: {token}")

                return {'error': 'Unauthorized'}, 401

            question = escape(request.json.get("question"))
            stream = request.json.get("stream", True)
            session_id = request.json.get("session_id")
            self.logger.debug(f"ask - email: {email} - question: "
                              f"{question}")
            save_message(email, "user", question)

            output = self.ask(question, stream=stream)

            return self.answer(output, email, stream=stream)

        @self.app.route('/send-otp', methods=['POST', 'OPTIONS'])
        @cors_required
        def send_otp():
            """ Endpoint to send OTP to the specified email. """
            email = request.json.get('email')

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):

                return {'error': 'Invalid email format'}, 400

            otp = str(randint(100000, 999999))
            expiry = (datetime.now() + timedelta(minutes=15)).isoformat()
            self.otp_store[email] = {'otp': otp, 'expiry': expiry}

            try:
                response = send_email_otp(email, otp)
                self.save_storage()

                # FIXME : remove OTP code from logs
                # self.logger.debug(f"The OTP code {otp} is sent to {email} - "
                #                   f"status {response.status_code}")

                return {'message': 'OTP sent'}, 200

            except Exception as e:
                self.logger.error(f"Error occurred while sending OTP "
                                  f"{type(e)}: {e}")

                return {'error': 'Failed to send OTP'}, 500

        @self.app.route('/verify-otp', methods=['POST', 'OPTIONS'])
        @cors_required
        def verify_otp():
            """ Endpoint to verify OTP from the user. """
            email = request.json.get('email')
            otp = request.json.get('otp')

            if email not in self.otp_store:
                self.logger.error("error 404 - OTP not found")

                return {'error': 'OTP not found'}, 404

            elif str(datetime.now()) >= self.otp_store[email]['expiry']:
                self.logger.error("error 400 - OTP expired")
                del self.otp_store[email]

                return {'error': 'OTP expired'}, 400

            elif self.otp_store[email]['otp'] != otp:
                self.logger.error("error 400 - Incorrect OTP")

                return {'error': 'Incorrect OTP'}, 400

            # OTP verification succeeded
            self.logger.debug("OTP verification succeeded")

            # Remove OTP code
            # del self.otp_store[email]

            # Generate authentification token
            token = token_hex(16)
            self.session_tokens[email] = token
            self.save_storage()
            message = {
                'message': 'Verification succeeded',
                'token': token,
            }
            # FIXME : Remove this logs
            # self.logger.debug(f"Token generated {token}")

            return message, 200

    def answer(self, output, email, stream: bool = False):
        """ Formats and sends the chatbot's response.

        Parameters
        ----------
        output : str or generator of str
            Output of the LLM.
        stream : bool, optional
            If True then stream the LLM output, otherwise return the full output
            at the end of the generation (default).

        Returns
        -------
        Response
            Flask response of the LLM output.

        """
        if stream:
            def generator():
                ans = ""
                for chunk in output:
                    ans += chunk

                    yield chunk

                self.prompt_hist['assistant'] = ans
                self.logger.debug(f"ANSWER - {self.ai_name} : {ans}")
                save_message(email, "assistant", ans)

            response = Response(generator(), content_type='text/event-stream')

        else:
            self.prompt_hist['assistant'] = output
            self.logger.debug(f"ANSWER - {self.ai_name} : {output}")
            response = Response(output, content_type='text/plain')

        return response


if __name__ == "__main__":
    from config import ROOT
    import logging.config

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

    debug = True

    with MiniChatBotAPI(debug=debug) as app:
        app.run(host='0.0.0.0', port=5000, debug=debug)
