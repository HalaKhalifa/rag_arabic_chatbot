import os
import sys

mode = os.getenv("APP_MODE", "cli")

if mode == "backend":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage.py", "runserver", "0.0.0.0:8000"])
else:
    from ragchat.cli.chat_cli import app as chat_app
    import typer
    typer.run(chat_app)
