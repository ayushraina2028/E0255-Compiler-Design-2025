#!/bin/bash

# Take filename as input
filename=$1

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found!"
    exit 1
fi

# Check if the file is a .mlir file
if [[ $filename != *.mlir ]]; then
    echo "File is not a .mlir file!"
    exit 1
fi

# get filename without extension
filename_no_ext="${filename%.*}"

mlir-opt --convert-arith-to-llvm --convert-func-to-llvm $filename -o LowerLevelIR.mlir
mlir-translate --mlir-to-llvmir LowerLevelIR.mlir -o $filename_no_ext.ll
rm LowerLevelIR.mlir