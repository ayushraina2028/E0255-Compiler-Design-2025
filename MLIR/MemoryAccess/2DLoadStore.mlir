module {
    func.func @memory_access2(%arg0 : memref<10x10xf32>) -> () {

        affine.for %i = 1 to 10 step 1 {
            affine.for %j = 1 to 10 step 1 {

                %val = affine.load %arg0[%i, %j] : memref<10x10xf32>
                %new_val = arith.addf %val, %val : f32
                affine.store %new_val, %arg0[%i, %j] : memref<10x10xf32>

            }
        }

        return
    }
}