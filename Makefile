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

status:
	@if ps aux | grep "[g]unicorn.*wsgi:app" > /dev/null 2>&1; then \
		echo "Backend server is running."; \
		echo "Process Details:"; \
		ps -f -C gunicorn; \
	else \
		echo "Backend server is not running."; \
	fi

stop:
	@echo "Stopping backend server..."
	pkill -f gunicorn

update:
	@echo "Updating GitHub repository..."
	git pull origin main
	cd PyLLMSol && git pull origin main && cd ..
	cd llama.cpp && git pull && cd ..
# 	@echo "Updating Python dependencies..."
# 	$(ACTIVATE) && pip install --upgrade -r ~/MiniChatBot/requirements.txt

clean:
	@echo "Cleaning virtual env..."
	rm -rf ~/venv
	@echo "Cleaning logs..."
	rm -rf src/logs/*

help:
	@echo "Available commands:"
	@echo "  setup         - Install requirements and setup virtual env and Nginx"
	@echo "  install_gpu   - Install GPU drivers and CUDA"
	@echo "  train         - Run training pipeline"
	@echo "  run           - Start the backend server"
	@echo "  status        - Check backend server status"
	@echo "  stop          - Stop the backend server"
	@echo "  update        - Pull latest code and update dependencies"
	@echo "  clean         - Clean up environment and logs"
