; Generated from a switch statement.

; CHECK: @switch
define dso_local i32 @switch(i32 noundef %0, ptr noundef %1) #0 {
  switch i32 %0, label %11 [
    i32 0, label %3
    i32 1, label %3
    i32 2, label %7
    i32 3, label %7
  ]

  ; CHECK: mul
  ; CHECK: add
  ; CHECK: add
  ; CHECK-NOT: mul
  ; CHECK-NOT: mul
  ; CHECK: ret

3:                                                ; preds = %2, %2
  %4 = mul i32 %0, %0
  %5 = add i32 %4, %0
  %6 = add i32 %5, 3
  br label %15

7:                                                ; preds = %2, %2
  %8 = mul i32 %0, %0
  %9 = add i32 %8, %0
  %10 = add i32 %9, 3
  br label %15

11:                                               ; preds = %2
  %12 = mul i32 %0, %0
  %13 = add i32 %12, %0
  %14 = add i32 %13, 3
  br label %15

15:                                               ; preds = %11, %7, %3
  %16 = phi i32 [ %14, %11 ], [ %10, %7 ], [ %6, %3 ]
  ret i32 %16
}