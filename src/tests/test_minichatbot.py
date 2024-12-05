#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-05 08:52:50
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 11:16:20
# @File path: ./src/tests/test_minichatbot.py
# @Project: MiniChatBot

""" Test MiniChatBot object. """

# Built-in packages
from pathlib import Path
from unittest.mock import patch

# Third party packages
from pyllmsol.data.chat import Chat
from pyllmsol.tests.mock import MockLlama
import pytest

# Local packages
from minichatbot import MiniChatBot
from config import PROMPT_PATH

__all__ = []


@pytest.fixture
def chatbot():
    """Fixture to initialize a MiniChatBot instance with mock parameters."""
    with patch("minichatbot.Llama", MockLlama):
        bot = MiniChatBot(verbose=True, n_ctx=1024, n_threads=2)

    return bot


@pytest.fixture
def prompt(chatbot):
    return Chat.from_jsonl(
        PROMPT_PATH / "short_prompt.jsonl",
        tokenizer=chatbot.llm.tokenizer(),
    )


def test_chatbot_initialization(chatbot, prompt):
    """Test initialization of MiniChatBot."""
    # Assert initial attributes
    assert chatbot.ai_name == "Assistant"
    assert chatbot.verbose is True
    assert chatbot.user_name == "User"
    assert chatbot.stop == "<|eot_id|>"
    # Validate the initial and history prompts match the expected prompt
    assert chatbot.init_prompt.text == prompt.text
    assert chatbot.prompt_hist.text == prompt.text
    # Check that the PromptFactory is correctly set
    assert chatbot.PromptFactory == Chat


def test_set_init_prompt(chatbot, prompt):
    """Test setting an initial prompt from a JSONL file."""
    # Test valid JSONL prompt
    chatbot.set_init_prompt(PROMPT_PATH / "short_prompt.jsonl")
    assert chatbot.init_prompt.text == prompt.text

    # Test valid JSON prompt
    chatbot.set_init_prompt(PROMPT_PATH / "short_prompt.json")
    assert chatbot.init_prompt.text == prompt.text

    # Test invalid file format
    with pytest.raises(ValueError, match="Unvalable file format, must be JSON or JSONL."):
        chatbot.set_init_prompt(Path("/mock/path/to/invalid.txt"))


if __name__ == "__main__":
    pass