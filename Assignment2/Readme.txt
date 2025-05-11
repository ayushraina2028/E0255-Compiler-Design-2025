Affine Loop Interchange Pass in MLIR

To see the implementation of the pass, check out LoopInterchange.cpp in the current directory.  
If you want to run my implementation, download this patch file from [this link](https://drive.google.com/file/d/1cgLEG6fK0ZPnihi-wKENf2RRfQR7HBoh/view?usp=sharing). You need to switch to git version of LLVM project commit id
b9d27ac252265839354fffeacaa8f39377ed7424 (Mar 17, 2025) as the base version to apply the patch

After applying the patch file, we can use:
1. cd build
2. ninja mlir-opt 

to build the pass.

Running the pass
1. build/bin/mlir-opt testcase.mlir -affine-loop-interchange 

Running with FileCheck
1. build/bin/mlir-opt testcase.mlir -affine-loop-interchange | FileCheck testcase.mlir

Running with Debug Statements
1. build/bin/mlir-opt testcase.mlir -affine-loop-interchange -debug-only=affine-loop-interchange 

Current Implementation passes 10/10 Given test cases.

Some Limitations: 
1. I have used some techniques like detecting column major access for imperfect nest case instead of 
full dependence analysis, otherwise code was getting a lot more complicated.

2. I have used AI model to generate one function which interchanges the imperfect nest, however techniques in previous point
I have implemented on my own.

However this analysis is enough to pass the given test cases.
Thanks