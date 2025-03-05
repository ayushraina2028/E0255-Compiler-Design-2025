#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a number as an argument"
    echo "Usage: $0 <number>"
    exit 1
fi  

# Step 1: Generate LLVM IR from C code
clang -Xclang -disable-O0-optnone -S -emit-llvm test$1.c -o test$1.ll

# Step 2: Apply mem2reg and save the output separately
opt -passes="mem2reg" test$1.ll -S -o test$1_mem2reg.ll

# Step 3: Run your custom pass on the mem2reg-transformed IR
opt -load-pass-plugin=./build/libHoistAnticipatedExpressions.so -passes="hoist-anticipated-expressions" test$1_mem2reg.ll -o -disable-output
