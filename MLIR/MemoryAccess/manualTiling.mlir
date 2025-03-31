module {
    func.func @tiledVersion(%A : memref<16xf32>) -> () {

        affine.for %ii = 0 to 16 step 4 {
            affine.for %i = 0 to 4 step 1 {

                %index = affine.apply affine_map<(a,b) -> (a+b)> (%ii, %i)
                %value = affine.load %A[%index] : memref<16xf32>
                %new_value = arith.addf %value, %value : f32
                affine.store %new_value, %A[%index] : memref<16xf32> 

            }
        }
        return
    }
}