module {
    func.func @nested_For_loop() -> () {
        affine.for %i = 0 to 10 step 1 {
            affine.for %j = 0 to 10 step 1 {
                %val = arith.addi %i, %j : index
            }
        }
        return
    }
}