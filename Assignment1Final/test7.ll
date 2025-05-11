; if/else with multiple redundant expressions in the block.

; Function Attrs: nounwind uwtable
; CHECK-LABEL: @if_else_multiple_redundant_exprs
define dso_local i32 @if_else_multiple_redundant_exprs(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp ugt i32 %0, 2
  br i1 %3, label %4, label %9

  ; There should be only one mul left
  ; CHECK: mul
  ; CHECK-NOT: mul
  ; CHECK: ret

4:                                                ; preds = %2
  %5 = mul i32 %0, %0
  %6 = add i32 %5, %0
  %7 = add i32 %6, 5
  %8 = add i32 %7, 5
  br label %14

9:                                                ; preds = %2
  %10 = mul i32 %0, %0
  %11 = add i32 %10, %0
  %12 = add i32 %11, 3
  %13 = mul i32 %0, %0
  br label %14

14:                                               ; preds = %9, %4
  %15 = phi i32 [ %8, %4 ], [ %13, %9 ]
  %16 = phi i32 [ %7, %4 ], [ %12, %9 ]
  %17 = add i32 %16, %15
  ret i32 %17
}