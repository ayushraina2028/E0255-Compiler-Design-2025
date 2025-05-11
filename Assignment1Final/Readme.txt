# **LLVM Pass: Hoist Anticipated Expressions**  

This project implements an **LLVM pass** that **computes anticipated expressions** using **dataflow analysis** and **performs code hoisting** to optimize the LLVM IR.  

To see the implementation of the pass, check out HoistAnticipatedExpressions.cpp in the current directory.  
If you want to run my implementation, download this patch file from [this link](https://drive.google.com/file/d/1y0yA3KBFoRKWWLMSBy7ScyqaZYJjtTMq/view?usp=sharing). You need to switch to LLVM git repository at tag llvmorg-19.1.7 (Jan 12, 2025) to apply this patch.

Follow these steps after applying the patch:

## **Building the Pass**  
To build the pass, run the following commands in the root directory of the project:  

1. cd build
2. ninja opt 

## **Running the Pass**
To run the pass on an LLVM IR file, run the following command:  
1. build/bin/opt -passes=hoist-anticipated-expressions TestFile.ll | FileCheck TestFile.ll

If the above command gives no error then we have passed the test case, and in similar way the pass can be tested for different test cases. Current implementation passes 8/9 given test cases.