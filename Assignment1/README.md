# **LLVM Pass: Hoist Anticipated Expressions**  

This project implements an **LLVM pass** that **computes anticipated expressions** using **dataflow analysis** and **performs code hoisting** to optimize the LLVM IR.  

## **Building the Pass**  
To build the pass, run the following commands in the root directory of the project:  

```sh
mkdir build
cd build
cmake ..
make
cd ..


## **Running the Pass**
To run the pass on an LLVM IR file, run the following command:  

opt -load-pass-plugin=./build/libHoistAnticipatedExpressions.so -passes="hoist-anticipated-expressions" test1.ll -o -disable-output
cat modified_output.ll

1. Replace test1.ll with any other test file from the Test Cases folder.
2. The pass modifies the input LLVM IR and generates an output file named modified_output.ll, containing the transformed IR after performing code hoisting.

## **Handling Multiple Test Files**
1. If the pass is run multiple times, the output will be overwritten in modified_output.ll
