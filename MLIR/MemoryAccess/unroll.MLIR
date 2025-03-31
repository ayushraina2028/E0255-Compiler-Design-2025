module {
    func.func @loop_without_unroll(%A : memref<8xf32>) -> () {
        affine.for %i = 0 to 8 step 1 {

            %value = affine.load %A[%i] : memref<8xf32>
            %new_value = arith.addf %value, %value : f32
            affine.store %new_value, %A[%i] : memref<8xf32>

        }
        return
    }
}

module {
    func.func @loop_with_unroll(%A : memref<8xf32>) -> () {
        affine.for %i = 0 to 8 step 2 {

            %value1 = affine.load %A[%i] : memref<8xf32>
            %new_value1 = arith.addf %value1, %value1 : f32
            affine.store %new_value1, %A[%i] : memref<8xf32>

            %value2 = affine.load %A[%i+1] : memref<8xf32>
            %new_value2 = arith.addf %value2, %value2 : f32
            affine.store %new_value2, %A[%i + 1] : memref<8xf32>
        }

        return
    }
}