#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a number as an argument"
    echo "Usage: $0 <number>"
    exit 1
fi

clang -Xclang -disable-O0-optnone -S -emit-llvm test$1.c -o test$1.ll
opt -load-pass-plugin=./build/libInstructionLevelAnalysis.so -passes=instruction-level-analysis test$1.ll -o -disable-output