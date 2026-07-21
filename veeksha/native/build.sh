#!/usr/bin/env bash
# Rebuild the veeksha_native C++ extension in place, for development.
#
# You do NOT normally need this: `pip install veeksha` (or `pip install -e .`)
# builds the extension as part of the install, on every platform, and degrades
# gracefully to the Python main loop if no C++ toolchain is present.
#
# This is the fast path for iterating on native_loop.cpp without a
# reinstall. It delegates to the SAME setuptools build the install uses, so
# the compiler, standard and flags cannot drift from what ships.
#
# Usage: veeksha/native/build.sh [python]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${1:-python}"

cd "$ROOT"
"$PY" setup.py build_ext --inplace
