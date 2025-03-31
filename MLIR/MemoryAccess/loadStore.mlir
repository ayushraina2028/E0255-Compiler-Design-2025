module {
    func.func @memory_access(%arg0 : memref<10xf32>) -> () {
        affine.for %i = 0 to 10 step 1 {
            %val = affine.load %arg0[%i] : memref<10xf32>
            %new_val = arith.addf %val, %val : f32
            affine.store %new_val, %arg0[%i] : memref<10xf32>
        }
        return
    }
}