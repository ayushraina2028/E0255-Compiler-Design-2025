// Test for handling other than add/mul.

// CHECK-LABEL: func @interchange_for_spatial_locality_mod
func.func @interchange_for_spatial_locality_mod(%A: memref<2048x2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %v = affine.load %A[%i mod 2, %k, %j] : memref<2048x2048x2048xf64>
        affine.store %v, %A[%i mod 2, %k, %j] : memref<2048x2048x2048xf64>
        // Interchanged for spatial locality.
        // CHECK:       affine.load %arg0[%{{.*}} mod 2, %arg1, %arg3] : memref<2048x2048x2048xf64>
        // CHECK-NEXT:  affine.store %0, %arg0[%{{.*}} mod 2, %arg1, %arg3] : memref<2048x2048x2048xf64>
      }
    }
  }
  return
}
