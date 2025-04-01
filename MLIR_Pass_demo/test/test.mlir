module {
  func.func @test() {
    affine.for %i = 0 to 10 {
      %val = arith.addi %i, %i : index
    }
    return
  }
}
