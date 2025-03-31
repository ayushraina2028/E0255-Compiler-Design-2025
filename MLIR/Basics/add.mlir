module {
    func.func @add(%arg0 : i32, %arg1 : i32) -> i32 {
        %sum = arith.addi %arg0, %arg1 : i32
        return %sum : i32
    }

    func.func @main() -> i32 {
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %result = call @add(%c1, %c2) : (i32, i32) -> i32
        return %result : i32 
    }
}