module {
    func.func @loop_without_fusion(%A : memref<8xf32>, %B : memref<8xf32>) -> () {

        affine.for %i = 0 to 8 {
            
            %value = affine.load %A[%i] : memref<8xf32>
            %new_value = arith.addf %value, %value : f32
            affine.store %new_value, %A[%i] : memref<8xf32>

        }

        affine.for %i = 0 to 8 {
            
            %value = affine.load %A[%i] : memref<8xf32>
            %new_value = arith.mulf %value, %value : f32
            affine.store %new_value, %B[%i] : memref<8xf32>

        }

        return
    }
}

// Use this command to run generate fused code:
// mlir-opt --affine-loop-fusion Fusion.mlir