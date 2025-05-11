// CHECK-LABEL: func @interchange_for_spatial_temporal
func.func @interchange_for_spatial_temporal(%A: memref<2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%j] : memref<2048xf64>
      affine.load %A[%j] : memref<2048xf64>
      affine.load %A[%i] : memref<2048xf64>
    }
  }
  return
}

// More reuse with %j, %i order.
// CHECK:       affine.load %arg0[%arg1] : memref<2048xf64>
// CHECK-NEXT:  affine.load %arg0[%arg1] : memref<2048xf64>
// CHECK-NEXT:  affine.load %arg0[%arg2] : memref<2048xf64>