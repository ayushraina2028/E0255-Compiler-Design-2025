// CHECK-LABEL: func @test_group_reuse
func.func @test_group_reuse(%A: memref<2048x2048xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%i, %j] : memref<2048x2048xf64>
      affine.store %v, %C[%i, %j] : memref<?x?xf64>
      %u1 = affine.load %A[%j, %i] : memref<2048x2048xf64>
      %u2 = affine.load %A[%j - 1, %i] : memref<2048x2048xf64>
      %u3 = affine.load %A[%j + 1, %i] : memref<2048x2048xf64>
      %s1 = arith.addf %u1, %u2 : f64
      %s2 = arith.addf %s1, %u3 : f64
      affine.store %s2, %B[%j, %i] : memref<?x?xf64>
    }
  }
  return
}

// Interchanged for better reuse.
// CHECK:      affine.for %[[I:.*]] =
// CHECK-NEXT:   affine.for %[[J:.*]] =
// CHECK:          affine.store %{{.*}}, %{{.*}}[%[[J]], %[[I]]] : memref<?x?xf64>
// CHECK:          affine.store %{{.*}}, %{{.*}}[%[[I]], %[[J]]] : memref<?x?xf64>