; ModuleID = 'test9.ll'
source_filename = "test9.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @not_anticipated_switch(i32 noundef %0, ptr noundef %1) #0 {
  switch i32 %0, label %10 [
    i32 0, label %3
    i32 1, label %3
    i32 2, label %3
  ]

3:                                                ; preds = %2, %2, %2
  %4 = srem i32 %0, 2
  %5 = srem i32 %0, 2
  %6 = mul nsw i32 %4, %5
  %7 = srem i32 %0, 2
  %8 = add nsw i32 %6, %7
  %9 = add nsw i32 %8, 3
  br label %14

10:                                               ; preds = %2
  %11 = mul nsw i32 %0, %0
  %12 = add nsw i32 %11, %0
  %13 = add nsw i32 %12, 2
  br label %14

14:                                               ; preds = %10, %3
  %.0 = phi i32 [ %13, %10 ], [ %9, %3 ]
  ret i32 %.0
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 6a3007683bf2fa05989c12c787f5547788d09178)"}
