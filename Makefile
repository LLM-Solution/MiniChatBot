install:
	bash setup.py

run:
	source ~/venv/bin/activate
	gunicorn --log-config ~/MiniChatBot/logging.ini --chdir ~/MiniChatBot/src --bind 0.0.0.0:5000 wsgi:app --timeout 120

clean:
	rm -rf ~/venv ~/MiniChatBot