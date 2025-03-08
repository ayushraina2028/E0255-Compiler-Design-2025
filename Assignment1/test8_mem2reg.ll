; CHECK: @not_anticipated_for_loop
define dso_local i32 @not_anticipated_for_loop(i32 noundef %0, ptr noundef %1) {
  br label %3

3:                                                ; preds = %8, %2
  %4 = phi i32 [ undef, %2 ], [ %10, %8 ]
  %5 = phi i32 [ 0, %2 ], [ %11, %8 ]
  %6 = icmp ult i32 %5, 10
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  ret i32 %4

  ; CHECK: mul
  ; CHECK-NEXT: srem
  ; CHECK-NEXT: add
  ; CHECK-NEXT: br label %{{.*}}, !llvm.loop

8:                                                ; preds = %3
  %9 = mul nsw i32 %0, %0
  %10 = srem i32 %9, %0
  %11 = add i32 %5, 1
  br label %3
}
