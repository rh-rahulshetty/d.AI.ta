import click

from streamlit.web import cli


@click.command()
def main():
    """Start streamlit server."""
    cli.main_run(["src/app.py"])
