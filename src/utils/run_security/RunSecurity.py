"""
Detect and display Python virtual environment status.

This module demonstrates understanding of Python virtual environments by:
- Detecting whether code runs in a venv or global Python
- Displaying environment-specific information
- Providing setup instructions for virtual environments
"""

import tomllib
import re
import site
import sys
import os
import time
from importlib import metadata


class RunEnvironmentError(Exception):
    """Raised when the runtime environment check fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"[EnvironmentError] {self.message}"


class RunSecurity:
    """Checks the runtime environment for security and correct setup."""

    def __init__(self) -> None:
        self.__virtual_env: str | None = None
        self.__sys_prefix: bool = False
        self.__dependencies: list[str] = []

    def check_process(self) -> None:
        """
        Main process to check environment and dependencies.

        Verifies that the script runs in a virtual environment and
        validates that all required dependencies are installed.

        Raises:
            RunEnvironmentError: If no venv is detected or dependencies
            are missing.
        """

        self.__get_sys_infos()
        try:
            if self.__sys_prefix and not self.__virtual_env:
                self.__display_global_env_warning(self.__virtual_env)
                raise RunEnvironmentError("No virtual environment detected.")
            else:
                self.__display_venv_info(self.__virtual_env)
                self.__get_dependencies()
                check, line = self.__check_dependencies(self.__dependencies)
                if not check:
                    raise RunEnvironmentError("Missing required dependencies.")
                time.sleep(0.2)
                sys.stdout.write(f"\033[{line + 13}A\033[J")
                sys.stdout.flush()
        except RunEnvironmentError:
            raise
        except Exception as e:
            raise RunEnvironmentError(f"Unexpected error: {e}") from e

    def __get_sys_infos(self) -> None:
        self.__virtual_env = os.environ.get("VIRTUAL_ENV")
        self.__sys_prefix = sys.prefix == sys.base_prefix

    def __get_dependencies(self) -> None:
        try:
            with open("pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            self.__dependencies = data["project"].get("dependencies", [])
        except FileNotFoundError:
            pass

    def __display_venv_info(self, virtual_env: str | None) -> None:
        """
        Display virtual environment information and success message.

        When running inside a virtual environment, this function shows:
        - Current Python executable path
        - Virtual environment name and path
        - Location of site-packages directory
        - Security confirmation of isolated environment

        Args:
            virtual_env: Path to the active virtual environment
            from VIRTUAL_ENV
        """
        print("\nVENV STATUS: Active\n")
        print("Python executable:", sys.executable)
        venv_name: str = (
            os.path.basename(virtual_env) if virtual_env else "unknown"
        )
        print(f"Environment name : {venv_name}")
        print(f"Environment path : {virtual_env}\n")
        print(
            "Isolated environment detected.\n"
            "Packages will not affect the global system.\n"
        )

        print("Site-packages path:")
        if virtual_env:
            site_packages_path: str = os.path.join(
                virtual_env,
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            )
            print(site_packages_path)
        else:
            for s in site.getsitepackages():
                print(s)

    def __display_global_env_warning(self, virtual_env: str | None) -> None:
        """
        Display global environment warning and setup instructions.

        When running in the global Python environment (no venv detected),
        this function warns the user about the security risks and provides
        clear instructions for creating and activating a virtual environment.

        Args:
            virtual_env: Value of VIRTUAL_ENV (should be None in global env)
        """
        print("\nVENV STATUS: No virtual environment detected\n")
        print("Python executable:", sys.executable)
        print(f"VIRTUAL_ENV: {virtual_env}")
        print(
            "WARNING: Running in the global Python environment.\n"
            "Installing packages here may affect your entire system.\n"
        )
        print(
            "To set up a virtual environment, run:\n"
            "  uv venv\n"
            "  source .venv/bin/activate  # Unix\n"
            "  .venv\\Scripts\\activate    # Windows\n"
        )
        print("Then restart the program.")

    def __check_dependencies(self, module_list: list[str]) -> tuple[bool, int]:
        """
        Verify that all required packages are installed.

        Iterates through module_list, attempting to import each package
        and retrieve its version from package metadata.

        Args:
            module_list: List of package names from pyproject.toml dependencies

        Returns:
            tuple[bool, int]: A pair of ``(all_loaded, lines_written)`` where
                ``all_loaded`` is ``True`` if every dependency was found and
                ``lines_written`` is the number of console lines printed
                (used by the caller to clear the output).
        """
        print("Checking dependencies...\n")
        print("Required packages:")
        all_loaded: bool = True
        lines_written: int = 3

        for raw_name in module_list:
            package_name: str = re.split(r"[><=!;\s\[]", raw_name)[0].strip()
            try:
                meta = metadata.metadata(package_name)
                print(f"\r    [OK] {meta['Name']} {meta['Version']} - found")
                lines_written += 1
            except metadata.PackageNotFoundError:
                all_loaded = False
                print(
                    f"\r    [MISSING] '{package_name}' could not be found. "
                    "Run 'uv sync' to install missing dependencies."
                )
                lines_written += 1

        return all_loaded, lines_written
