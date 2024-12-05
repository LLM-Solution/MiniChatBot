#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-05 11:28:44
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 11:48:23
# @File path: ./src/tests/test_utils.py
# @Project: MiniChatBot

""" Test util functions. """

# Built-in packages
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime
from json import JSONDecodeError, load, loads, dumps

# Third party packages
from cryptography.fernet import Fernet
import pytest
from sendgrid.helpers.mail import Mail
from sendgrid import SendGridAPIClient

# Local packages
from utils import (
    load_storage,
    save_storage,
    get_env_variables,
    send_email_otp,
    save_message,
)

__all__ = []


@pytest.fixture
def mock_env_file():
    """Fixture to create a temporary .env file."""
    with NamedTemporaryFile("w", delete=False) as temp_file:
        temp_file.write("STORAGE_KEY=mock_storage_key\nSENDGRID_API_KEY=mock_sendgrid_key\n")
        temp_file_path = temp_file.name
    yield Path(temp_file_path)
    Path(temp_file_path).unlink()


@pytest.fixture
def mock_storage_file():
    """Fixture to create a temporary storage file."""
    with NamedTemporaryFile("wb", delete=False) as temp_file:
        temp_file_path = temp_file.name
    yield Path(temp_file_path)
    Path(temp_file_path).unlink()


@pytest.fixture
def mock_history_dir():
    """Fixture to create a temporary directory for conversation history."""
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_load_storage(mock_env_file, mock_storage_file):
    """Test loading storage with mock encryption."""
    # Prepare mock environment and storage data
    with patch("utils.ENV_PATH", mock_env_file), \
         patch("utils.STORAGE_PATH", mock_storage_file):
        key = Fernet.generate_key()
        with patch("utils.get_env_variables", return_value=key.decode()):
            fernet = Fernet(key)
            mock_data = {"otp_store": {}, "session_tokens": {}}
            encrypted_data = fernet.encrypt(dumps(mock_data).encode())
            mock_storage_file.write_bytes(encrypted_data)

            # Test loading storage
            result = load_storage()
            assert result == mock_data


def test_save_storage(mock_env_file, mock_storage_file):
    """Test saving storage with mock encryption."""
    with patch("utils.ENV_PATH", mock_env_file), \
         patch("utils.STORAGE_PATH", mock_storage_file):
        key = Fernet.generate_key()
        with patch("utils.get_env_variables", return_value=key.decode()):
            data_to_save = {"otp_store": {}, "session_tokens": {}}
            save_storage(data_to_save)

            # Verify saved data
            encrypted_data = mock_storage_file.read_bytes()
            fernet = Fernet(key)
            decrypted_data = loads(fernet.decrypt(encrypted_data).decode())
            assert decrypted_data == data_to_save


def test_get_env_variables(mock_env_file):
    """Test retrieving environment variables."""
    with patch("utils.ENV_PATH", mock_env_file):
        value = get_env_variables("STORAGE_KEY")
        assert value == "mock_storage_key"

        with pytest.raises(KeyError):
            get_env_variables("NON_EXISTENT_KEY")


def test_send_email_otp(mock_env_file):
    """Test sending an OTP email."""
    with (patch("utils.ENV_PATH", mock_env_file),
          patch.object(SendGridAPIClient, "send", return_value="mock_response") as mock_send):
        response = send_email_otp("test@example.com", "123456")
        assert response == "mock_response"
        mock_send.assert_called_once()
        message = mock_send.call_args[0][0]
        assert isinstance(message, Mail)
        assert str(message.subject) == "Code verification"
        assert "123456" in message._contents[0].content


def test_save_message(mock_history_dir):
    """Test saving a message to conversation history."""
    # Create a temporary directory for mock conversation history
    with patch("utils.CONV_HISTORY_PATH", mock_history_dir):
        save_message("test_user", "user", "Hello, world!")
        history_file = mock_history_dir / "test_user.json"
        assert history_file.exists()

        # Verify saved content
        with history_file.open("r", encoding="utf-8") as f:
            saved_data = load(f)
        assert len(saved_data) == 1
        assert saved_data[0]["message"] == "Hello, world!"
        assert saved_data[0]["role"] == "user"


if __name__ == "__main__":
    pass
