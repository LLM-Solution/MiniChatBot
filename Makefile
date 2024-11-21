.PHONY setup install_gpu train run clean

setup:
	bash setup.sh

install_gpu:
	sudo bash install_gpu.sh

train:
	@echo "Starting training pipeline..."
	bash train_model.sh

run:
	source ~/venv/bin/activate
	gunicorn --log-config ~/MiniChatBot/logging.ini --chdir ~/MiniChatBot/src --bind 0.0.0.0:5000 wsgi:app --timeout 120

clean:
	rm -rf ~/venv
	rm -rf src/logs/*