#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-24 23:42:59
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-30 08:49:00

""" Description. """

# Built-in packages

# Third party packages

# Local packages
from api import MiniChatBotAPI

__all__ = []

# Create MiniChatBotAPI instance
api_instance = MiniChatBotAPI()

# Expose app attribute for Gunicorn
app = api_instance.app
