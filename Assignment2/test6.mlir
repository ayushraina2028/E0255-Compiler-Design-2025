// CHECK-LABEL: func @interchange_invalid
func.func @interchange_invalid(%A: memref<2048x2048xf64>) {
  affine.for %t = 0 to 2048 {
    affine.for %i = 0 to 2048 {
      %u1 = affine.load %A[%t - 1, %i] : memref<2048x2048xf64>
      %u2 = affine.load %A[%t - 1, %i + 1] : memref<2048x2048xf64>
      %u3 = affine.load %A[%t - 1, %i - 1] : memref<2048x2048xf64>
      %s1 = arith.addf %u1, %u2 : f64
      %s2 = arith.addf %s1, %u3 : f64
      affine.store %s2, %A[%t, %i] : memref<2048x2048xf64>
    }
  }
  return
}

// Interchange is invalid.
// CHECK: affine.store %{{.*}}[%arg1, %arg2] : memref<2048x2048xf64>