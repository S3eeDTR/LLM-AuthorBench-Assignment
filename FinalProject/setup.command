#!/bin/bash
set -eu
cd "$(dirname "$0")"
if command -v python3.12 >/dev/null 2>&1; then PROJECT_PYTHON=python3.12
elif command -v python3.11 >/dev/null 2>&1; then PROJECT_PYTHON=python3.11
else PROJECT_PYTHON=python3
fi
"$PROJECT_PYTHON" -c 'import sys; assert (3,11)<=sys.version_info[:2]<=(3,12), "Install Python 3.11 or 3.12 from python.org, then run setup again."'
"$PROJECT_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-mac.txt
printf '\nSetup complete. Run start.command next.\n'
