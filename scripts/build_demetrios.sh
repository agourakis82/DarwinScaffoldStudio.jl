#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMETRIOS_DIR="${ROOT_DIR}/demetrios"
COMPILER_DIR="${DEMETRIOS_DIR}/compiler"

if [ ! -d "${COMPILER_DIR}" ]; then
    echo "Demetrios submodule not found at ${DEMETRIOS_DIR}."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is required to build Demetrios."
    exit 1
fi

echo "Building Demetrios compiler..."
cargo build --release --manifest-path "${COMPILER_DIR}/Cargo.toml"

echo ""
echo "Set environment for CompilerBridge:"
echo "export DEMETRIOS_HOME=\"${DEMETRIOS_DIR}\""
echo "export DEMETRIOS_STDLIB=\"${DEMETRIOS_DIR}/stdlib\""
