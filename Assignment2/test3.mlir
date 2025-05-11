// CHECK-LABEL: func.func @matmul_ijk
func.func @matmul_ijk(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %a = affine.load %A[%i, %k] : memref<2048x2048xf64>
        %b = affine.load %B[%k, %j] : memref<2048x2048xf64>
        %ci = affine.load %C[%i, %j] : memref<2048x2048xf64>
        %p = arith.mulf %a, %b : f64
        %co = arith.addf %ci, %p : f64
        affine.store %co, %C[%i, %j] : memref<2048x2048xf64>
      }
    }
  }
  return
}

// Test whether the ikj permutation has been found.

// CHECK:      affine.load %arg0[%arg3, %arg4] : memref<2048x2048xf64>
// CHECK-NEXT: affine.load %arg1[%arg4, %arg5] : memref<2048x2048xf64>
// CHECK-NEXT: affine.load %arg2[%arg3, %arg5] : memref<2048x2048xf64>
// CHECK-NEXT: arith.mulf %0, %1 : f64
// CHECK-NEXT: arith.addf %2, %3 : f64
// CHECK-NEXT: affine.store %4, %arg2[%arg3, %arg5] : memref<2048x2048xf64>