module {
    func.func @affine_map_example(%arg0 : index) -> (index) {

        %result = affine.apply affine_map<(d) -> (d * 2)> (%arg0)
        return %result : index

    }
}

module {
    func.func @affine_map_example2(%arg0 : memref<10xf32>) -> () {

        affine.for %i = 0 to 5 step 1 {

            %index = affine.apply affine_map<(d) -> (3 * d)> (%i)
            %value = affine.load %arg0[%index] : memref<10xf32>
            %new_value = arith.addf %value, %value : f32
            affine.store %new_value, %arg0[%index] : memref<10xf32>

        }

        return 
    }
}

// 2D map
module {
    func.func @affine_map_example3(%arg0 : memref<10x10xf32>, %offset : index) -> () {

        affine.for %i = 0 to 5 step 1 {
            %index = affine.apply affine_map<(a,b) -> (a+b)> (%i, %offset) 
            %value = affine.load %arg0[%index, %i] : memref<10x10xf32>
        }

        return 

    }
}