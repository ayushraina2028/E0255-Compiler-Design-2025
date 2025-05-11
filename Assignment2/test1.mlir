// CHECK-LABEL: func @interchange_for_spatial_locality
func.func @interchange_for_spatial_locality(%A: memref<2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%j, %i] : memref<2048x2048xf64>
      affine.store %v, %A[%j, %i] : memref<2048x2048xf64>
    }
  }
  return
}

// Interchanged for spatial locality.
// CHECK:       affine.load %arg0[%arg1, %arg2] : memref<2048x2048xf64>
// CHECK-NEXT:  affine.store %{{.*}}, %arg0[%arg1, %arg2] : memref<2048x2048xf64>
