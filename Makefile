SHELL := /bin/bash
.PHONY: setup install_gpu train run clean

VENV_PATH = ~/venv
ACTIVATE = source $(VENV_PATH)/bin/activate

setup:
	@echo "Installing dependencies, set virtual env and setup Nginx..."
	./setup.sh

install_gpu:
	@echo "Starting installing GPU drivers and CUDA..."
	sudo ./install_gpu.sh

train:
	@echo "Starting training pipeline..."
	./train_model.sh

run:
	@echo "Starting running backend server..."
	$(ACTIVATE) && gunicorn --log-config ~/MiniChatBot/logging.ini --chdir ~/MiniChatBot/src --bind 0.0.0.0:5000 wsgi:app --timeout 120

stop:
	@echo "Stopping backend server..."
	pkill -f "gunicorn --log-config ~/MiniChatBot/logging.ini"

clean:
	@echo "Cleaning virtual env..."
	rm -rf ~/venv
	@echo "Cleaning logs..."
	rm -rf src/logs/*

update:
	@echo "Updating GitHub repository..."
	git pull origin main
	cd PyLLMSol && git pull origin main && cd ..
	cd llama.cpp && git pull origin main && cd ..
	@echo "Updating Python dependencies..."
	$(ACTIVATE) && pip install --upgrade -r ~/MiniChatBot/requirements.txt

help:
	@echo "Available commands:"
	@echo "  setup         - Install requirements and setup virtual env and Nginx"
	@echo "  install_gpu   - Install GPU drivers and CUDA"
	@echo "  train         - Run training pipeline"
	@echo "  run           - Start the backend server"
	@echo "  stop          - Stop the backend server"
	@echo "  clean         - Clean up environment and logs"
	@echo "  update        - Pull latest code and update dependencies"
