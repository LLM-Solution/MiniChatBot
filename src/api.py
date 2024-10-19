#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-18 17:26:54
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-18 18:05:18

""" Flask API object. """

# Built-in packages
from config import MODEL_NAME

# Third party packages
from flask import Flask, request, jsonify, make_response
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local packages

__all__ = []


app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Models are loaded")


@app.route('/minichatbot', methods=['POST', 'OPTIONS'])
def minichatbot():
    if request.method == 'OPTIONS':
        # Réponse aux requêtes OPTIONS pour gérer CORS (pré-vol)
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")

        return response

    user_input = request.json.get('message')
    length = len(tokenizer(user_input).input_ids)
    print(f"User: {user_input}")

    encoded = tokenizer(user_input, return_tensors='pt')
    response = model.generate(**encoded, max_length=length + 32)
    decoded = tokenizer.decode(response[0])
    print(f"MiniChatBot: {decoded}")

    response = make_response(jsonify({'response': decoded}))
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
