"""Entry point for the Call Me Maybe application.

Handles environment bootstrapping, argument parsing, and delegates
execution to the Controller. When run directly (not as a child process),
spawns a new terminal window and re-launches itself as a child.
"""

import logging
import os
import shutil
import subprocess
import sys
import time

from src.utils.logger.Logger import setup_logger
from src.utils.run_security import RunSecurity, RunEnvironmentError


PROG_NAME: str = "RAG against the machine"
PROG_DESCRIPTION: str = "What the program does"  # a faire
PROG_HELP: str = "Text at the bottom of help"  # a faire


def get_terminal_command():
    known_terminals = [
        {"cmd": "gnome-terminal", "flag": "--"},
        {"cmd": "konsole", "flag": "-e"},
        {"cmd": "xfce4-terminal", "flag": "-e"},
        {"cmd": "alacritty", "flag": "-e"},
        {"cmd": "terminator", "flag": "-x"},
        {"cmd": "xterm", "flag": "-e"},
    ]
    for term in known_terminals:
        if shutil.which(term["cmd"]):
            return [term["cmd"], term["flag"]]

    return None


def main() -> None:
    """Run the main application flow.

    Sets up the runtime security check, parses CLI arguments, loads the
    LLM and its supporting components, then starts the generation loop
    via the Controller.  On any unrecoverable error the user is prompted
    to press Enter before the process exits.
    """
    logger = logging.getLogger(PROG_NAME)
    secure_env = RunSecurity()
    try:
        secure_env.check_process()
        time.sleep(0.2)
    except RunEnvironmentError as e:
        logger.error(f"{e}")
        logger.info("Programm exit")
        input("\n\nPress Enter to exit...")
        return
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        input("\n\nPress Enter to exit...")
        return

    input("\n\nPress Enter to exit...")
    return


if __name__ == "__main__":
    if "--child" not in sys.argv and "--gui" not in sys.argv:
        terminal = get_terminal_command()
        args = [sys.executable, "-m", "src", "--child"] + sys.argv[1:]
        subprocess.Popen(
            [terminal[0], terminal[1]] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        os._exit(0)
    setup_logger(PROG_NAME)
    main()
