#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-05 10:21:42
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 11:16:50
# @File path: ./src/tests/test_api.py
# @Project: MiniChatBot

""" Test MiniChatBotAPI object. """

# Built-in packages
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Third party packages
from flask import Flask, Response
from pyllmsol.tests.mock import MockLlama
import pytest

# Local packages
from api import MiniChatBotAPI
from config import GGUF_MODEL

__all__ = []

@pytest.fixture
def minichatbot():
    """Fixture to create a MiniChatBotAPI instance."""
    with patch("minichatbot.Llama", MockLlama):
        return MiniChatBotAPI(debug=True, verbose=True, n_ctx=1024, n_threads=2)

@pytest.fixture()
def app(minichatbot):
    return minichatbot.app


@pytest.fixture
def client(app):
    """Fixture to provide a test client for the Flask app."""
    return app.test_client()


def test_load_storage(minichatbot):
    """Test the load_storage method."""
    mock_storage = {
        "otp_store": {"test@example.com": {"otp": "123456", "expiry": "2024-12-06T12:00:00"}},
        "session_tokens": {"test@example.com": "valid_token"}
    }
    with patch("api.load_storage", return_value=mock_storage):
        minichatbot.load_storage()
        assert minichatbot.otp_store == mock_storage["otp_store"]
        assert minichatbot.session_tokens == mock_storage["session_tokens"]


def test_save_storage(minichatbot):
    """Test the save_storage method."""
    minichatbot.otp_store = {"test@example.com": {"otp": "123456", "expiry": "2024-12-06T12:00:00"}}
    minichatbot.session_tokens = {"test@example.com": "valid_token"}
    with patch("api.save_storage") as mock_save:
        minichatbot.save_storage()
        mock_save.assert_called_once_with({
            "otp_store": minichatbot.otp_store,
            "session_tokens": minichatbot.session_tokens,
        })


def test_answer_stream(minichatbot):
    """Test the answer method with streaming enabled."""
    email = "test@example.com"
    chunks = ["Hello", " ", "world", "!"]
    output = (chunk for chunk in chunks)  # Simulate generator output
    with patch("api.save_message") as mock_save_message:
        response = minichatbot.answer(output, email, stream=True)
        assert isinstance(response, Response)
        assert response.content_type == "text/event-stream"
        # Test response streaming
        response_data = "".join(response.response)# .decode()
        assert response_data == "".join(chunks)
        mock_save_message.assert_called_once_with(email, "assistant", "".join(chunks))


def test_answer_non_stream(minichatbot):
    """Test the answer method without streaming."""
    email = "test@example.com"
    output = "Hello, world!"
    with patch("api.save_message") as mock_save_message:
        response = minichatbot.answer(output, email, stream=False)
        assert isinstance(response, Response)
        assert response.content_type == "text/plain"
        assert response.data.decode() == output
        mock_save_message.assert_called_once_with(email, "assistant", output)


def test_health_check(client):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200


def test_send_otp_valid_email(client):
    """Test sending OTP with a valid email."""
    with patch("api.send_email_otp", return_value=MagicMock(status_code=200)):
        response = client.post("/send-otp", json={"email": "test@example.com"})
        assert response.status_code == 200
        assert response.json["message"] == "OTP sent"


def test_send_otp_invalid_email(client):
    """Test sending OTP with an invalid email."""
    response = client.post("/send-otp", json={"email": "invalid-email"})
    assert response.status_code == 400
    assert response.json["error"] == "Invalid email format"


def test_verify_otp_success(client, minichatbot):
    """Test verifying OTP with correct credentials."""
    email = "test@example.com"
    otp = "123456"
    token = "mock_token"
    expiry = (datetime.now() + timedelta(minutes=15)).isoformat()
    minichatbot.otp_store[email] = {"otp": otp, "expiry": expiry}
    with patch("api.token_hex", return_value=token):
        response = client.post("/verify-otp", json={"email": email, "otp": otp})
        assert response.status_code == 200
        assert response.json["message"] == "Verification succeeded"
        assert response.json["token"] == token


def test_verify_otp_incorrect(client, minichatbot):
    """Test verifying OTP with an incorrect OTP."""
    email = "test@example.com"
    minichatbot.otp_store[email] = {"otp": "123456", "expiry": (datetime.now() + timedelta(minutes=15)).isoformat()}
    response = client.post("/verify-otp", json={"email": email, "otp": "654321"})
    assert response.status_code == 400
    assert response.json["error"] == "Incorrect OTP"


def test_verify_otp_expired(client, minichatbot):
    """Test verifying OTP when the OTP has expired."""
    email = "test@example.com"
    minichatbot.otp_store[email] = {"otp": "123456", "expiry": (datetime.now() - timedelta(minutes=1)).isoformat()}
    response = client.post("/verify-otp", json={"email": email, "otp": "123456"})
    assert response.status_code == 400
    assert response.json["error"] == "OTP expired"


def test_ask_authorized(client, minichatbot):
    """Test asking a question with a valid token."""
    email = "test@example.com"
    question = "What is the weather today?"
    token = "valid_token"
    minichatbot.session_tokens[email] = token
    with patch.object(minichatbot, "ask", return_value="Sunny"):
        with patch("api.save_message"):
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post("/ask", json={"email": email, "question": question}, headers=headers)
            assert response.status_code == 200
            assert response.data.decode("utf-8") == "Sunny"


def test_ask_unauthorized(client, minichatbot):
    """Test asking a question with an invalid token."""
    email = "test@example.com"
    token = "invalid_token"
    minichatbot.session_tokens[email] = "valid_token"
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/ask", json={"email": email, "question": "Hello"}, headers=headers)
    assert response.status_code == 401
    assert response.json["error"] == "Unauthorized"

if __name__ == "__main__":
    pass
