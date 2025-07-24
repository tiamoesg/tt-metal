#!/bin/bash
set -eo pipefail

# Colors
GREEN="\e[32m"
YELLOW="\e[33m"
PURPLE="\e[35m"
CYAN="\e[36m"
RESET="\e[0m"

# Check if the script is sourced
is_sourced=false
if [ "$0" != "$BASH_SOURCE" ]; then
    is_sourced=true
fi

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo -e "${CYAN}Using user-specified Python: $PYTHON_CMD${RESET}"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo -e "${YELLOW}Python command not found: $PYTHON_CMD${RESET}"
    exit 1
fi

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo -e "${CYAN}Creating virtual environment in: $PYTHON_ENV_DIR${RESET}"

# Create virtual environment
$PYTHON_CMD -m venv $PYTHON_ENV_DIR

# Activate if sourced
if [ "$is_sourced" = true ]; then
    source $PYTHON_ENV_DIR/bin/activate
    echo -e "${GREEN}Activated source as python_env.${RESET}"
else
    echo -e "${YELLOW}[Note] Virtual environment created but not activated in current shell.${RESET}"
    echo -e "${CYAN}To activate, run:${RESET} source $PYTHON_ENV_DIR/bin/activate"
fi

echo -e "${CYAN}Forcefully using a version of pip that will work with our view of editable installs${RESET}"
$PYTHON_ENV_DIR/bin/pip install --force-reinstall pip==21.2.4

echo -e "${CYAN}Setting up virtual environment${RESET}"
$PYTHON_ENV_DIR/bin/python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
$PYTHON_ENV_DIR/bin/python3 -m pip install setuptools wheel==0.45.1

echo -e "${CYAN}Installing dev dependencies${RESET}"
$PYTHON_ENV_DIR/bin/python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt

echo -e "${CYAN}Installing tt-metal${RESET}"
$PYTHON_ENV_DIR/bin/pip install -e .

# Add environment variables to venv activation script
echo -e "${GREEN}Configuring TT-Metal environment variables in venv activation...${RESET}"
cat <<EOL >> $PYTHON_ENV_DIR/bin/activate

# TT-Metal environment variables
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

echo -e "${PURPLE}[Reminder:${RESET} ${BLUE}If you switch card types (e.g., wormhole_b0 → blackhole), update ARCH_NAME in python_env/bin/activate${RESET}]"
EOL


# Do not install hooks when this is a worktree
if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
    echo -e "${CYAN}Generating git hooks${RESET}"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo -e "${YELLOW}In worktree: not generating git hooks${RESET}"
fi

echo -e "${GREEN}Successfully created venv as python_env.${RESET}"

if [ "$is_sourced" = false ]; then
    echo -e "${CYAN}To activate, run:${RESET} source $PYTHON_ENV_DIR/bin/activate"
fi

# Prompt for stubs only if venv is active
if [ "$is_sourced" = true ]; then
    echo -e "${PURPLE}Would you like to generate stubs now?${RESET}"
    echo -e "  1) Yes (default)"
    echo -e "  2) No"
    read -p "Select an option [1/2]: " choice
    choice=${choice:-1}
    if [ "$choice" = "1" ]; then
        echo -e "${CYAN}Generating stubs...${RESET}"
        ./scripts/build_scripts/create_stubs.sh || echo -e "${YELLOW}[Warning] Stub generation failed. You can try later with:${RESET} ./scripts/build_scripts/create_stubs.sh"
    else
        echo -e "${YELLOW}Skipped stub generation. You can run it later with:${RESET} ./scripts/build_scripts/create_stubs.sh"
    fi
else
    echo -e "${CYAN}Happy coding! - If you want stubs, run:${RESET} ./scripts/build_scripts/create_stubs.sh"
fi
