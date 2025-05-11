// Test for interchange on imperfect nests.

// CHECK-LABEL: func @imperfect_nest
func.func @imperfect_nest(%A: memref<2048x2048xf64>) {
  %c0 = arith.constant 0.0 : f64
  %c1 = arith.constant 1.0 : f64
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 1024 {
      affine.store %c0, %A[%j, %i] : memref<2048x2048xf64>
    }
    affine.for %j = 1024 to 2048 {
      affine.store %c1, %A[%j, %i] : memref<2048x2048xf64>
    }
  }
  return
}
// CHECK:      for %{{.*}} = 0 to 1024 {
// CHECK-NEXT:   for %{{.*}} = 0 to 2048 {
// CHECK-NEXT:     affine.store %{{.*}}, %arg0[%arg1, %arg2]
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: for %{{.*}} = 1024 to 2048 {
// CHECK-NEXT:   for %{{.*}} = 0 to 2048 {
// CHECK-NEXT:      affine.store %{{.*}}, %arg0[%arg1, %arg2]
// CHECK-NEXT:   }
// CHECK-NEXT: }