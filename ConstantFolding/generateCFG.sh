#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a number as an argument"
    echo "Usage: $0 <number>"
    exit 1
fi

# Move DOT files from temp directory
mv /tmp/cfg.*.dot .

# Generate PNG with numbered filename
dot -Tpng cfg.*.dot -o CFG_Graph$1.png

# Optional: Clean up DOT files
rm cfg.*.dot