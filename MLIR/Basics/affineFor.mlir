module {
    func.func @loop_example() -> () {
        affine.for %i = 0 to 10 step 1 {
            %val = arith.addi %i, %i : index
        }
        return
    }
}