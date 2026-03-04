# Prerequisites

## Pyenv

```
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libgdbm-dev \
    libnss3-dev
```

`curl -fsSL https://pyenv.run | bash`

Add the following to /.bashrc:

```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Reload shell: `source ~/.bashrc`

Install python: `pyenv install 3.12.3`

# Build

Move to project folder, then:

```
pyenv local 3.12  # sets 3.12 for this project
python --version     # should show 3.12.x
```

```
python -m venv .venv
source .venv/bin/activate
python --version   # should be 3.12.x
```

## Create application

Install build tools: `pip install --upgrade pip setuptools wheel build`

Build wheel: `python -m build`

Install cli: `pip install ./dist/cli_second_brain-0.1.0-py3-none-any.whl`

# Installation

Set pythoh version to global access: `pyenv global 3.12`

Install pipx: `pip install --user pipx`

Restart terminal, then: `pipx ensurepath`

Install CLI:

`pipx install git+ssh://git@github-cli_second_brain/jlad26/second_brain_cli.git`