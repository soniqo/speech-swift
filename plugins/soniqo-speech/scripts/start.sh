#!/bin/bash
set -euo pipefail
plugin_root="$(cd "$(dirname "$0")/.." && pwd)"
node_binary="${SPEECH_PLUGIN_NODE:-}"
if [[ -z "$node_binary" ]]; then
  for candidate in /opt/homebrew/bin/node /usr/local/bin/node; do
    if [[ -x "$candidate" ]]; then
      node_binary="$candidate"
      break
    fi
  done
fi
if [[ -z "$node_binary" ]]; then
  node_binary="$(command -v node || true)"
fi
if [[ -z "$node_binary" || ! -x "$node_binary" ]]; then
  echo 'Soniqo Speech requires Node.js 22 or later. Install it with: brew install node' >&2
  exit 1
fi
if ! "$node_binary" -e 'if (Number(process.versions.node.split(".")[0]) < 22) process.exit(1)'; then
  echo 'Soniqo Speech requires Node.js 22 or later.' >&2
  exit 1
fi
exec "$node_binary" "$plugin_root/dist/server.cjs"
