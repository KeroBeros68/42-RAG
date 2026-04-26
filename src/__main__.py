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

from src.Controller import Controller
from src.utils.logger.Logger import setup_logger
from src.utils.run_security import RunSecurity, RunEnvironmentError


PROG_NAME: str = "RAG against the machine"
PROG_DESCRIPTION: str = (
    "RAG against the machine — hybrid code retrieval pipeline. "
    "Indexes a codebase with BM25 and semantic embeddings, "
    "then retrieves relevant source passages for natural language queries."
)

PROG_HELP: str = (
    "Commands:\n"
    "  index          Build the retrieval index from raw sources\n"
    "  search         Retrieve top-k passages for a query\n"
    "  search_dataset Run retrieval over a full question dataset\n"
    "\n"
    "Examples:\n"
    "  python -m src index --max_chunk_size=1500 --chroma=True\n"
    "  python -m src search 'how does the cache work' --k=10\n"
    "  python -m src search_dataset --path=code --k=5\n"
    "\n"
    "Run 'python -m src <command> --help' for command-specific options.\n"
    "Dependencies must be installed: make install"
)


def get_terminal_command() -> list[str] | None:
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

    if "--child" in sys.argv:
        sys.argv.remove("--child")

    from fire import Fire
    try:
        Fire(Controller(logger))
    except SystemExit as e:
        if e.code != 0:
            logger.error(f"Fire error (Code: {e.code})")
            logger.info("Program exit")
        else:
            pass
    except Exception as e:
        logger.error(f"{str(e)}")
        logger.info("Programm exit")
        input("\n\nPress Enter to exit...")
        return
    input("\n\nPress Enter to exit...")
    return


if __name__ == "__main__":
    if "--child" not in sys.argv and "--gui" not in sys.argv:
        terminal = get_terminal_command()
        if not terminal:
            pass
        else:
            args = [sys.executable, "-m", "src", "--child"] + sys.argv[1:]
            subprocess.Popen(
                [terminal[0], terminal[1]] + args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                ),
            )
            os._exit(0)
    setup_logger(PROG_NAME)
    main()
