; Generated from an if/else.

; CHECK-LABEL: @simple_if_else
define dso_local i32 @simple_if_else(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp ugt i32 %0, 2
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = mul i32 %0, %0
  %6 = add i32 %5, %0
  %7 = add i32 %6, 5
  br label %12
  ; Only one instance of mul + add + add should be left.
  ; CHECK: %[[M:.*]] = mul i32
  ; CHECK: add {{.*}} %[[M]]
  ; CHECK: add
  ; CHECK-NOT: mul
  ; CHECK-NOT: add
  ; CHECK: ret

8:                                                ; preds = %2
  %9 = mul i32 %0, %0
  %10 = add i32 %9, %0
  %11 = add i32 %10, 5
  br label %12

12:                                               ; preds = %8, %4
  %13 = phi i32 [ %7, %4 ], [ %11, %8 ]
  ret i32 %13
}
