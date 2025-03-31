module {
  func.func @tiling_example(%A: memref<16xf32>) {
    affine.for %i = 0 to 16 {
      %val = affine.load %A[%i] : memref<16xf32>
      %new_val = arith.addf %val, %val : f32
      affine.store %new_val, %A[%i] : memref<16xf32>
    }
    return
  }
}


// This command automatically generates tiled code: 
// mlir-opt --affine-loop-tile=tile-size=4 tile.mlir

