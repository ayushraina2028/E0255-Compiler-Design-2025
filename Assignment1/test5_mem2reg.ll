; ModuleID = 'test5.ll'
source_filename = "test5.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @switch_case(i32 noundef %0, ptr noundef %1) #0 {
  switch i32 %0, label %11 [
    i32 0, label %3
    i32 1, label %3
    i32 2, label %7
    i32 3, label %7
  ]

3:                                                ; preds = %2, %2
  %4 = mul nsw i32 %0, %0
  %5 = add nsw i32 %4, %0
  %6 = add nsw i32 %5, 3
  br label %15

7:                                                ; preds = %2, %2
  %8 = mul nsw i32 %0, %0
  %9 = add nsw i32 %8, %0
  %10 = add nsw i32 %9, 3
  br label %15

11:                                               ; preds = %2
  %12 = mul nsw i32 %0, %0
  %13 = add nsw i32 %12, %0
  %14 = add nsw i32 %13, 3
  br label %15

15:                                               ; preds = %11, %7, %3
  %.0 = phi i32 [ %14, %11 ], [ %10, %7 ], [ %6, %3 ]
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
