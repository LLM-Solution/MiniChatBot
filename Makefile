setup:
	bash setup.py

install_gpu:
	sudo bash install_gpu.sh

run:
	source ~/venv/bin/activate
	gunicorn --log-config ~/MiniChatBot/logging.ini --chdir ~/MiniChatBot/src --bind 0.0.0.0:5000 wsgi:app --timeout 120

clean:
	rm -rf ~/venv ~/MiniChatBot