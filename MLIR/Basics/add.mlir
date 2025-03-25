module {
    func.func @add(%arg0 : i32, %arg1 : i32) -> i32 {
        %sum = arith.addi %arg0, %arg1 : i32
        return %sum : i32
    }
}