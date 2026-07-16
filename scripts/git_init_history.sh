#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "docker-compose.yml" || ! -d "warehouse" ]]; then
  echo "Run this script from the CryptoStream repository root." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init -q
  git checkout -q -b main 2>/dev/null || git branch -q -M main
fi

echo "This helper script is kept here for future history replay work."
