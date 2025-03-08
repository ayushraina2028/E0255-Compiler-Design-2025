; A multiple if/else if/else block with anticipated expressions.

; CHECK-LABEL: @simple_if_else_multiple
define dso_local i32 @simple_if_else_multiple(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp ugt i32 %0, 3
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = mul i32 %0, %0
  %6 = add i32 %5, %0
  %7 = add i32 %6, 5
  br label %18
  ; Only one instance of mul + add + add should be left.
  ; CHECK: mul
  ; CHECK: add
  ; CHECK: add
  ; CHECK-NOT: mul
  ; CHECK-NOT: add
  ; CHECK: ret

8:                                                ; preds = %2
  %9 = icmp ugt i32 %0, 2
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  %11 = mul i32 %0, %0
  %12 = add i32 %11, %0
  %13 = add i32 %12, 5
  br label %18

14:                                               ; preds = %8
  %15 = mul i32 %0, %0
  %16 = add i32 %15, %0
  %17 = add i32 %16, 5
  br label %18

18:                                               ; preds = %10, %14, %4
  %19 = phi i32 [ %7, %4 ], [ %13, %10 ], [ %17, %14 ]
  ret i32 %19
}