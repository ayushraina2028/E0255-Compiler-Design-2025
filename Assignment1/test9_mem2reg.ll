; Not anticipated expression here.
; CHECK: @not_anticipated_switch
define dso_local i32 @not_anticipated_switch(i32 noundef %0, ptr noundef %1) {
  switch i32 %0, label %8 [
    i32 0, label %3
    i32 1, label %3
    i32 2, label %3
  ]
  ; CHECK: urem
  ; CHECK-NEXT: mul
  ; CHECK-NEXT: add
  ; CHECK-NEXT: add


  ; CHECK: mul
  ; CHECK-NEXT: add
  ; CHECK-NEXT: add

3:                                                ; preds = %2, %2, %2
  %4 = urem i32 %0, 2
  %5 = mul i32 %4, %4
  %6 = add i32 %5, %4
  %7 = add i32 %6, 3
  br label %12

8:                                                ; preds = %2
  %9 = mul i32 %0, %0
  %10 = add i32 %9, %0
  %11 = add i32 %10, 2
  br label %12

12:                                               ; preds = %8, %3
  %13 = phi i32 [ %11, %8 ], [ %7, %3 ]
  ret i32 %13
}