#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-05 18:23:21
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-09 18:42:39

""" Util functions. """

# Built-in packages
from datetime import datetime
from json import loads, dumps, JSONDecodeError

# Third party packages
from cryptography.fernet import Fernet
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Local packages
from config import ENV_PATH, STORAGE_PATH, CONV_HISTORY_PATH

__all__ = []


def load_storage():
    """ Load storage for OTP code and tokens.

    Returns
    -------
    dict
        Loaded storage with key "otp_store" and "session_tokens".

    """
    try:
        key = get_env_variables("STORAGE_KEY")
        fernet = Fernet(key)

        with STORAGE_PATH.open('rb') as f:

            encrypted_data = f.read()

        return loads(fernet.decrypt(encrypted_data).decode())

    except (FileNotFoundError, JSONDecodeError):

        return {"otp_store": {}, "session_tokens": {}}


def save_storage(data):
    """ Save storage for OTP code and tokens.

    Parameters
    ----------
    data : dict
        Storage data.

    """
    key = get_env_variables("STORAGE_KEY")
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(dumps(data).encode())

    with STORAGE_PATH.open("wb") as f:
        f.write(encrypted_data)


def get_env_variables(name: str) -> str:
    """ Get variable from `.env`.

    Parameters
    ----------
    name : str
        Name of the requested variable.

    Returns
    -------
    str
        Value of the loaded variable.

    Raises
    ------
    KeyError
        If the requested variable is not found in the `.env` file.

    """
    with ENV_PATH.open("r") as env_file:
        for line in env_file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)

                if key == name:

                    return value

        else:

            raise KeyError(f"The variable {name} does not exist in {ENV_PATH}.")


def send_email_otp(email, otp):
    """ Send an OTP verification code by email.

    Parameters
    ----------
    email : str
        Email address to send the OTP code verification.
    otp : str
        OTP code verification to send.

    """
    sendgrid_api_key = get_env_variables("SENDGRID_API_KEY")
    subject = "Code verification"
    content = (f"Hello,\n\nYour OTP code verification is {otp}.\nThis code is "
               f"available for 15 minutes.\n\nBest regards,\nLLM Solutions")
    message = Mail(
        from_email="no-reply@llm-solutions.fr",
        to_emails=email,
        subject=subject,
        plain_text_content=content,
    )
    sg = SendGridAPIClient(sendgrid_api_key)

    return sg.send(message)


def save_message(identifiant: str, role: str, message: str):
    path = CONV_HISTORY_PATH / f"{identifiant}.json"

    if path.exists():
        conv = loads(path.read_text(encoding='utf-8'))

    else:
        conv = []

    conv.append({
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "message": message,
    })

    path.write_text(dumps(conv), encoding='utf-8')


if __name__ == "__main__":
    pass
