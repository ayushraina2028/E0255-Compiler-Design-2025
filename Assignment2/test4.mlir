// CHECK-LABEL: func.func @interchange_for_outer_parallelism
func.func @interchange_for_outer_parallelism(%A: memref<2048x2048x2048xf64>) {
  affine.for %i = 1 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %v = affine.load %A[%i, %j, %k] : memref<2048x2048x2048xf64>
        %p = arith.mulf %v, %v : f64
        affine.store %p, %A[%i - 1, %j, %k] : memref<2048x2048x2048xf64>
      }
    }
  }
  return
}

// %j should become outermost - provides outer parallelism and locality.
// CHECK:      affine.load %arg0[%arg2, %arg1, %arg3]
// CHECK-NEXT: arith.mulf %0, %0 : f64
// CHECK-NEXT: affine.store %1, %arg0[%arg2 - 1, %arg1, %arg3]