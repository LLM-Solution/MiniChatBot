#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-23 16:25:55
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-06 10:09:11

""" Flask API object for MiniChatBot. """

# Built-in packages
from datetime import datetime, timedelta
from logging import getLogger
from markupsafe import escape
from random import randint
import re

# Third party packages
from flask import Flask, request, Response, jsonify

# Local packages
from _base_api import API, cors_required
from _base_cli import _BaseCommandLineInterface
from config import GGUF_MODEL, PROMPT
from utils import send_email_otp

__all__ = []


LOG = getLogger('app')


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

        self.otp_store = {}

        # Set API part
        LOG.debug("Start init Flask API object")
        self.app = Flask(__name__)
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
            question = escape(request.json.get("question"))
            stream = request.json.get("stream", True)
            session_id = request.json.get("session_id")
            LOG.debug(f"ask - session_id: {session_id} - question: {question}")

            output = self.ask(question, stream=stream)

            return self.answer(output, stream=stream)

        @self.app.route('/send-otp', methods=['POST', 'OPTIONS'])
        @cors_required
        def send_otp():
            """ Endpoint to send OTP to the specified email. """
            email = request.json.get('email')

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):

                return jsonify({'error': 'Invalid email format'}), 400

            otp = str(randint(100000, 999999))
            expiry = datetime.now() + timedelta(minutes=15)
            self.otp_store[email] = {'otp': otp, 'expiry': expiry}

            try:
                response = send_email_otp(email, otp)
                # FIXME : remove OTP code from logs
                self.logger.debug(f"The OTP code {otp} is sent to {email} - "
                                  f"status {response.status_code}")

                return jsonify({'message': 'OTP envoyé'}), 200

            except Exception as e:
                self.logger.error(f"Error occurred while sending OTP "
                                  f"{type(e)}: {e}")

                return jsonify({'error': 'Failed to send OTP'}), 500

        @self.app.route('/verify-otp', methods=['POST', 'OPTIONS'])
        @cors_required
        def verify_otp():
            """ Endpoint to verify OTP from the user. """
            email = request.json.get('email')
            otp = request.json.get('otp')

            if email not in self.otp_store:

                return jsonify({'error': 'OTP not found'}), 404

            elif datetime.now() >= self.otp_store[email]['expiry']:
                del self.otp_store[email]

                return jsonify({'error': 'OTP expired'}), 400
            
            elif self.otp_store[email]['otp'] != otp:

                return jsonify({'error': 'Incorrect OTP'}), 400

            else:
                del self.otp_store[email]

                return jsonify({'message': 'Verification succeeded'}), 200

    def answer(self, output, stream: bool = False):
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
                full_answer = ""
                for chunk in output:
                    full_answer += chunk
                    self.prompt_hist += chunk

                    yield chunk

                LOG.debug(f"answer - response: {full_answer}")

            response = Response(generator(), content_type='text/event-stream')

        else:
            self.prompt_hist += output
            LOG.debug(f"answer - response: {output}")
            response = Response(output, content_type='text/plain')

        return response

    def reset_prompt(self):
        """ Reset the current prompt history and load `init_prompt`. """
        LOG.debug("<Reset prompt>")
        self.prompt_hist = self.init_prompt + "\n"

        LOG.debug("<Load init prompt>")
        r = self.llm(f"{self.init_prompt}", max_tokens=1)

        LOG.debug(f"reset_prompt - output: {r}")


if __name__ == "__main__":
    from config import ROOT
    import logging.config

    # Load logging configuration
    logging.config.fileConfig(ROOT / 'logging.ini')

    debug = True

    with MiniChatBotAPI(debug=debug) as app:
        app.run(host='0.0.0.0', port=5000, debug=debug)
