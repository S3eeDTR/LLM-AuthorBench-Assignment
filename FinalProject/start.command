#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  printf 'Run setup.command first (or: bash setup.command).\n'
  exit 1
fi
.venv/bin/python start_mac.py
