#!/bin/bash
# Convenience wrapper for log2yaml tool
# This script runs the log2yaml Rust utility

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_PATH="$SCRIPT_DIR/log2yaml/target/release/log2yaml"

# Check if tool is built
if [ ! -f "$TOOL_PATH" ]; then
    echo "Error: log2yaml not built yet"
    echo "Building now..."
    cd "$SCRIPT_DIR/log2yaml" || exit 1
    cargo build --release || exit 1
    cd - > /dev/null || exit 1
fi

# Run the tool with all arguments passed through
"$TOOL_PATH" "$@"
