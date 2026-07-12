#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ./run.sh /path/to/photos [additional modal arguments]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="$1"
shift

if command -v modal >/dev/null 2>&1; then
  MODAL=(modal)
elif command -v uvx >/dev/null 2>&1; then
  MODAL=(uvx --from modal modal)
else
  echo "Modal is not installed. Install it or install uv so uvx is available." >&2
  exit 1
fi

"${MODAL[@]}" run "$SCRIPT_DIR/modal_app.py" --input-dir "$INPUT_DIR" "$@"
