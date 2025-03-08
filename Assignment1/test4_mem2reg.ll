; Generated from a simple for loop.

; CHECK-LABEL: @for_loop_invariant_expr
define dso_local i32 @for_loop_invariant_expr(i32 noundef %0, ptr noundef %1) #0 {
  br label %3

3:                                                ; preds = %9, %2
  %4 = phi i32 [ 0, %2 ], [ %12, %9 ]
  %5 = icmp ult i32 %4, 10
  br i1 %5, label %9, label %6

6:                                                ; preds = %3
  %7 = mul nsw i32 %0, %0
  %8 = srem i32 %7, %0
  ret i32 %8

9:                                                ; preds = %3
  %10 = mul nsw i32 %0, %0
  %11 = srem i32 %10, %0
  %12 = add i32 %4, 1
  br label %3
  ; CHECK: %[[M:.*]] = mul
  ; CHECK: srem i32 %[[M]], %{{.*}}
  ; CHECK-NOT: mul
  ; CHECK-NOT: srem
  ; CHECK: ret
}