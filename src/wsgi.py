#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-24 23:42:59
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-30 08:37:21

""" Description. """

# Built-in packages
import logging.config
import yaml

# Third party packages

# Local packages
from api import MiniChatBotAPI

__all__ = []

# Load logging configuration
with open('./logging.ini', 'r') as f:
    log_config = yaml.safe_load(f.read())

logging.config.dictConfig(log_config)

# Create MiniChatBotAPI instance
api_instance = MiniChatBotAPI()

# Expose app attribute for Gunicorn
app = api_instance.app
