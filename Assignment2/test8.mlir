// Test to make sure there are no crashes/aborts on things that aren't handled.

// CHECK-LABEL: func @if_else
func.func @if_else(%A: memref<2048x2048xf64>) {
  %c0 = arith.constant 0.0 : f64
  %c1 = arith.constant 1.0 : f64
  affine.for %i = 0 to 2048 {
    affine.if affine_set<(d0) : (d0 - 1024 >= 0)> (%i) {
      affine.for %j = 0 to 2048 {
        affine.store %c0, %A[%i, %j] : memref<2048x2048xf64>
      }
    } else {
      affine.for %j = 0 to 2048 {
        affine.store %c1, %A[%i, %j] : memref<2048x2048xf64>
      }
    }
  }
  return
}