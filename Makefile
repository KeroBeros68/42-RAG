.SILENT:
.ONESHELL:
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# **************************************************************************** #
#                                   COLORS                                     #
# **************************************************************************** #

GREEN   := \033[0;32m
RED     := \033[0;31m
YELLOW  := \033[0;33m
BLUE    := \033[0;34m
MAGENTA := \033[0;35m
CYAN    := \033[0;36m
RESET   := \033[0m

ECHO    := echo -e

# **************************************************************************** #
#                                  VARIABLES                                   #
# **************************************************************************** #

NAME := main.py
VENV := .venv
PYTHON := $(VENV)/bin/python
INSTALL := uv
INSTALL_CMD := sync
PYTEST := $(VENV)/bin/pytest
TOML_FILE := pyproject.toml

VERSION := $(shell grep '^version' $(TOML_FILE) | sed 's/version = "\(.*\)"/\1/')
VERSION_MAJOR := $(word 1, $(subst ., ,$(VERSION)))
VERSION_MINOR := $(word 2, $(subst ., ,$(VERSION)))
VERSION_PATCH := $(word 3, $(subst ., ,$(VERSION)))

ARGV :=
SRC_MYPY ?= .
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

# **************************************************************************** #
#									.PHONY									   #
# **************************************************************************** #

.PHONY: help install run debug lint lint-strict clean fclean re \
		st add _commit push_feat push_fix push_refactor \
		push_docs push_style push_chore test version \
		bump-patch bump-minor bump-major push_release push_test

# **************************************************************************** #
#									Help								  	   #
# **************************************************************************** #

# Show available make commands.
help:
	$(ECHO) "$(YELLOW)Available commands:$(RESET)"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  📦 Setup$(RESET)"
	$(ECHO) "  make install              Install dependencies with $(INSTALL)"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  🚀 Run$(RESET)"
	$(ECHO) "  make run [ARGV=...]       Run the project (optional args)"
	$(ECHO) "  make debug [ARGV=...]     Run the project with pdb debugger"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  🧪 Tests$(RESET)"
	$(ECHO) "  make test                 Run tests with $(PYTEST)"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  🔍 Lint$(RESET)"
	$(ECHO) "  make lint                 Run flake8 + mypy"
	$(ECHO) "  make lint-strict          Run flake8 + mypy (strict mode)"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  🧹 Clean$(RESET)"
	$(ECHO) "  make clean                Remove __pycache__ and .mypy_cache"
	$(ECHO) "  make fclean               clean + remove app.log"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  🔧 Git$(RESET)"
	$(ECHO) "  make st                   git status"
	$(ECHO) "  make add                  git add all files"
	$(ECHO) "  make push_feat M='...'    Commit + push (type: feat)"
	$(ECHO) "  make push_fix M='...'     Commit + push (type: fix)"
	$(ECHO) "  make push_refactor M='...' Commit + push (type: refactor)"
	$(ECHO) "  make push_docs M='...'    Commit + push (type: docs)"
	$(ECHO) "  make push_style M='...'   Commit + push (type: style)"
	$(ECHO) "  make push_chore M='...'   Commit + push (type: chore)"
	$(ECHO) "  make push_test M='...'    Commit + push (type: test)"
	$(ECHO) "  make push_release M='...' Commit + push (type: feat + bump major)"
	$(ECHO) ""
	$(ECHO) "$(CYAN)  🏷 Versioning$(RESET)"

# **************************************************************************** #
#									Rules									   #
# **************************************************************************** #

# ###		APP RULES 		### #

# Install project dependencies.
install:
	$(ECHO) -n "$(CYAN)Checking $(INSTALL)...$(RESET) ";
	if command -v $(INSTALL) > /dev/null 2>&1; then
		$(ECHO) "$(GREEN)✓ $(INSTALL) is installed$(RESET)";
	else
		$(ECHO) "$(YELLOW)⚠ $(INSTALL) not found$(RESET)";
		$(ECHO) -n "$(CYAN)Installing $(INSTALL)...$(RESET) ";
		if python3 -m pip install --user --upgrade $(INSTALL) > /dev/null 2>&1; then
			$(ECHO) "$(GREEN)✓$(RESET)";
		else
			$(ECHO) "$(RED)✗ Failed to install $(INSTALL)$(RESET)";
			exit 1;
		fi;
	fi;
	$(ECHO) -n "$(CYAN)Installing dependencies with $(INSTALL)...$(RESET) ";
	if $(INSTALL) $(INSTALL_CMD) > /dev/null 2>&1; then
		$(ECHO) "$(GREEN)✓$(RESET)";
	else
		$(ECHO) "$(RED)✗$(RESET)";
		$(INSTALL) $(INSTALL_CMD);
	fi;
	$(ECHO) "$(GREEN)✓ Installation complete$(RESET)";
# Run the main program.
# $(PYTHON) $(NAME) $(ARGV)
run:
	$(PYTHON) -m src $(ARGV)

# Run the program with debugger.
debug:
	$(PYTHON) -m pdb $(NAME) $(ARGV)


# Reinstall from a clean state.
re: fclean install

# Run the test suite.
test:
	$(ECHO) "$(CYAN)Running tests...$(RESET)"
	if $(PYTEST); then
		$(ECHO) "$(GREEN)✓ All tests passed$(RESET)";
	else
		$(ECHO) "$(RED)✗ Tests failed$(RESET)";
		exit 1;
	fi


# ###		LINT RULES 		### #

# Run standard lint checks.
lint:
	$(ECHO) "$(CYAN)Running flake8...$(RESET)";
	$(FLAKE8) --exclude='.venv,llm_sdk, src/data'
	$(ECHO) "$(CYAN)Running mypy...$(RESET)";
	$(MYPY) --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '.venv' --exclude 'src/data' $(SRC_MYPY)
	$(MAKE) clean

# Run stricter lint checks.
lint-strict:
	$(ECHO) "$(CYAN)Running flake8...$(RESET)";
	$(FLAKE8) --exclude='.venv,llm_sdk,src/data'
	$(ECHO) "$(CYAN)Running mypy...$(RESET)";
	$(MYPY) --explicit-package-bases --strict --exclude '.venv' --exclude 'src/data' $(SRC_MYPY)
	$(MAKE) clean


# ###		CLEAN RULES 		### #

# Remove cache directories.
clean:
	$(ECHO) "$(CYAN)Suppression de __pycache__...$(RESET)"
	LIST_PYCACHE=$$(find . -type d -name "__pycache__" 2>/dev/null)
	if [ -n "$$LIST_PYCACHE" ]; then \
		find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
		$(ECHO) "$(GREEN)✓ Dossiers __pycache__ supprimés$(RESET)"; \
	else \
		$(ECHO) "$(YELLOW)⚠ Rien à nettoyer$(RESET)"; \
	fi
	
	$(ECHO) "$(CYAN)Suppression de .mypy_cache...$(RESET)"
	LIST_MYPY=$$(find . -type d -name ".mypy_cache" 2>/dev/null)
	if [ -n "$$LIST_MYPY" ]; then \
		find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; \
		$(ECHO) "$(GREEN)✓ Dossiers .mypy_cache supprimés$(RESET)"; \
	else \
		$(ECHO) "$(YELLOW)⚠ Rien à nettoyer$(RESET)"; \
	fi
	$(ECHO) "$(CYAN)Suppression de .pytest_cache...$(RESET)"
	LIST_PYTEST=$$(find . -type d -name ".pytest_cache" 2>/dev/null)
	if [ -n "$$LIST_PYTEST" ]; then \
		find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; \
		$(ECHO) "$(GREEN)✓ Dossiers .pytest_cache supprimés$(RESET)"; \
	else \
		$(ECHO) "$(YELLOW)⚠ Rien à nettoyer$(RESET)"; \
	fi

# Remove logs after cleaning.
fclean: clean
	$(ECHO) "$(CYAN)Suppression de app.log...$(RESET) "
	if find . -name "app.log" -type f | grep -q .; then
		find . -name "app.log" -type f -exec rm -f {} + 2>/dev/null;
		$(ECHO) "$(GREEN)✓ Fichier app.log supprimé$(RESET)";
	else
		$(ECHO) "$(YELLOW)⚠ Rien à nettoyer$(RESET)";
	fi


# ###		GIT RULES 		### #

# Show git working tree status.
st:
	git status

# Stage all current changes.
add:
	$(ECHO) "$(CYAN)--- Status ---$(RESET)"
	git status -s
	git add .
	$(ECHO) "$(GREEN)✔ Files added.$(RESET)"
	git status -s
	$(ECHO) "$(GREEN)✔ Done$(RESET)\n"

# Test, commit, and push changes.
_commit: add
	BRANCH=$$(git branch --show-current);
	if [ -z "$(M)" ]; then
		$(ECHO) "$(RED)Error: Le message du commit (M) est vide !$(RESET)";
		exit 1;
	fi
	git commit -m "$(TYPE): $(M)"
	git push --set-upstream origin $(BRANCH)
	$(ECHO) "$(GREEN)🚀 Successful push ($(TYPE)).$(RESET)\n"

# Commit and push a release.
push_release:
	$(MAKE) test && $(MAKE) bump-major && $(MAKE) _commit TYPE=feat M="$(M)"

# Commit and push a feature.
push_feat:
	$(MAKE) test && $(MAKE) bump-minor && $(MAKE) _commit TYPE=feat M="$(M)"

# Commit and push a fix.
push_fix:
	$(MAKE) test && $(MAKE) bump-patch && $(MAKE) _commit TYPE=fix M="$(M)"

# Commit and push a refactor.
push_refactor:
	$(MAKE) test && $(MAKE) _commit TYPE=refactor M="$(M)"

# Commit and push documentation.
push_docs:
	$(MAKE) test && $(MAKE) _commit TYPE=docs M="$(M)"

# Commit and push style changes.
push_style:
	$(MAKE) test && $(MAKE) _commit TYPE=style M="$(M)"

# Commit and push maintenance changes.
push_chore:
	$(MAKE) test && $(MAKE) _commit TYPE=chore M="$(M)"

# Commit and push test changes.
push_test:
	$(MAKE) test && $(MAKE) _commit TYPE=test M="$(M)"


# ###		VERSIONNING RULES 		### #

# Show current project version.
version:
	$(ECHO) "$(CYAN)Version: $(VERSION)$(RESET)"

# Increase patch version number.
bump-patch:
	$(eval NEW_PATCH := $(shell echo $$(($(VERSION_PATCH)+1))))
	$(eval NEW_VERSION := $(VERSION_MAJOR).$(VERSION_MINOR).$(NEW_PATCH))
	sed -i 's/^version = "$(VERSION)"/version = "$(NEW_VERSION)"/' $(TOML_FILE)
	$(ECHO) "$(GREEN)Version bumped: $(VERSION) → $(NEW_VERSION)$(RESET)"
# Increase minor version number.

bump-minor:
	$(eval NEW_MINOR := $(shell echo $$(($(VERSION_MINOR)+1))))
	$(eval NEW_VERSION := $(VERSION_MAJOR).$(NEW_MINOR).0)
	sed -i 's/^version = "$(VERSION)"/version = "$(NEW_VERSION)"/' $(TOML_FILE)
	$(ECHO) "$(GREEN)Version bumped: $(VERSION) → $(NEW_VERSION)$(RESET)"
# Increase major version number.

bump-major:
	$(eval NEW_MAJOR := $(shell echo $$(($(VERSION_MAJOR)+1))))
	$(eval NEW_VERSION := $(NEW_MAJOR).0.0)
	sed -i 's/^version = "$(VERSION)"/version = "$(NEW_VERSION)"/' $(TOML_FILE)
	$(ECHO) "$(GREEN)Version bumped: $(VERSION) → $(NEW_VERSION)$(RESET)"