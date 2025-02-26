#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a number as an argument"
    echo "Usage: $0 <number>"
    exit 1
fi

clang -Xclang -disable-O0-optnone -S -emit-llvm test$1.c -o test$1.ll
opt -load-pass-plugin=./build/libConstantFolding.so -passes="constant-folding" test$1.ll -S -o optimized_IR$1.ll