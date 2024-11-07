#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-05 18:23:21
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-06 00:02:46

""" Util functions. """

# Built-in packages

# Third party packages
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Local packages
from config import ROOT

__all__ = []


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
    path = ROOT / ".env"

    with path.open("r") as env_file:
        for line in env_file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)

                if key == name:

                    return value

        else:

            raise KeyError(f"The variable {name} does not exist in {path}.")


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


if __name__ == "__main__":
    pass
