#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ./view.sh /path/to/reconstruction [port]" >&2
  exit 2
fi

OUTPUT_DIR="$(cd "$1" && pwd)"
PORT="${2:-8008}"

if [[ ! -f "$OUTPUT_DIR/gaussians.splat" ]]; then
  echo "gaussians.splat not found in $OUTPUT_DIR" >&2
  exit 1
fi

URL="http://127.0.0.1:${PORT}/viewer.html?scene=gaussians.splat"
echo "Serving $OUTPUT_DIR"
echo "Open $URL"

if command -v open >/dev/null 2>&1; then
  (sleep 1 && open "$URL") &
fi

cd "$OUTPUT_DIR"
python3 -m http.server "$PORT" --bind 127.0.0.1
