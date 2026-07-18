#!/usr/bin/env bash
# Build the veeksha_native free-threaded C++ extension in place.
#
# Requires: a C++17 compiler and the free-threaded CPython + pybind11 whose
# `python` is on PATH (or passed as $1). Produces
# veeksha/native/veeksha_native<EXT_SUFFIX>.so next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${1:-python}"

PYINC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PBINC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
SUF="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="$HERE/veeksha_native$SUF"

echo "building $OUT"
clang++ -O2 -std=c++17 -shared -fPIC -undefined dynamic_lookup \
  -I"$PYINC" -I"$PBINC" \
  "$HERE/src/native_receiver.cpp" -o "$OUT"
echo "built $OUT"
