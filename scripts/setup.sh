#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (
# INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This function will make sure we are at the root of repo
function ensure_git_root {
    git_root="$(git rev-parse --show-toplevel 2>/dev/null)"
    if [ ! -d ".git" ]; then
        echo "Not at the root of the git repository."
        return 1  # Return failure if not at git root
    fi
    # Ask git if we are at the root
    if [ "$git_root" != "$PWD" ]; then
        echo "Not at the root of the git repository."
        echo "Please change directory to $git_root"
        return 1  # Return failure if not at git root
    fi
    return 0  # Return success if at git root
}

function check_pre_commit {
    # Check if pre-commit binary is available
    if(command -v pre-commit >/dev/null 2>&1); then
        return 0  # Return success if pre-commit is available
    else
        return 1  # Return failure if pre-commit is not available
    fi
}

function check_python3 {
    if(command -v python3 >/dev/null 2>&1); then
        return 0  # Return success if Python 3 is found
    else
        return 1  # Return failure if Python 3 is not found
    fi
}

function check_python3_version {
    # Version should be 3.8 or above
    version=$(python3 --version 2>&1 | awk '{print $2}')
    if [[ "$version" < "3.8" ]]; then
        return 1  # Return failure if version is below 3.8
    else
        return 0  # Return success if version is 3.8 or above
    fi
}

function check_venv_module {
    if (python3 -c "import venv" >/dev/null 2>&1); then
        return 0  # Return success if venv module is available
    else
        return 1  # Return failure if venv module is not available
    fi
}

function setup_pre_commit {
    # If any of the below commands error out, it will return 0
    python3 -m venv .venv || return 1
    source .venv/bin/activate || return 1
    pip install pre-commit || return 1
    pre-commit install || return 1
    return 0
}

function setup_docs {
    git worktree add docs/internal docs
}

function setup_exp {
    git worktree add exp experiments
}

function add_venv_to_shell_config {
    local venv_activate=$(realpath .venv/bin/activate)
    local shell_type=""
    local shell_config=""
    local activation_cmd="source $venv_activate"

    # Detect shell type
    if [[ -n "$BASH_VERSION" ]]; then
        shell_type="bash"
        shell_config="$HOME/.bashrc"
    elif [[ -n "$ZSH_VERSION" ]]; then
        shell_type="zsh"
        shell_config="$HOME/.zshrc"
    elif [[ $SHELL == *"fish"* ]]; then
        shell_type="fish"
        shell_config="$HOME/.config/fish/config.fish"
        activation_cmd="source $venv_activate.fish"
    else
        # Default to bash if we can't detect
        shell_type="bash"
        shell_config="$HOME/.bashrc"
    fi

    echo
    read -p "Would you like to add virtual environment activation to your $shell_type configuration ($shell_config)? (y/n): " add_to_config

    if [[ $add_to_config == "y" || $add_to_config == "Y" ]]; then
        # Add activation command to shell config
        if [[ $shell_type == "fish" ]]; then
            echo "if status is-interactive" >> "$shell_config"
            echo "    $activation_cmd" >> "$shell_config"
            echo "end" >> "$shell_config"
        else
            echo "" >> "$shell_config"
            echo "# Added by AOCL-DLP setup" >> "$shell_config"
            echo "$activation_cmd" >> "$shell_config"
        fi
        echo "Added virtual environment activation to $shell_config"
        echo "Please restart your shell or run: $activation_cmd"
    else
        echo
        echo "To activate the virtual environment manually, run:"
        echo "$activation_cmd"
    fi
}

# If we are not in git root, we need to exit with error
if ( ! ensure_git_root ); then
    exit 1
fi

# Check if the script has been already run
if [ -f ".setup_done" ]; then
    # Check if virtual environment is already active
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual environment is already active."
        echo "Cannot proceed, please deactivate!"
        exit 0;
    fi
    # Check if virtual environment exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
else
    if ( check_python3 ); then
        echo "Python 3 is installed."
    else
        echo "Python 3 is not installed. Please install it to continue."
        exit 1
    fi

    if ( check_venv_module ); then
        echo "venv module is available."
    else
        echo "If this machine is debian based, try sudo apt install python3-venv"
        echo "venv module is not available. Please install it to continue."
        exit 1
    fi

    if ( setup_pre_commit ); then
        echo "pre-commit has been set up successfully."
    else
        echo "Failed to set up pre-commit."
        exit 1
    fi

    if ( setup_docs ); then
        echo "Documentation setup has been completed successfully."
        echo "Directory docs/internal now points to docs branch"
    else
        echo "Failed to set up documentation."
        exit 1
    fi

    if ( setup_exp ); then
        echo "Experimentation setup has been completed successfully."
        echo "Directory exp now points to experiments branch"
    else
        echo "Failed to set up experimentation."
        exit 1
    fi

    # Prompt user to add venv activation to shell config
    add_venv_to_shell_config

    touch .setup_done

fi
