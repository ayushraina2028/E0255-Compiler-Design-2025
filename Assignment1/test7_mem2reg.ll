; ModuleID = 'test7.ll'
source_filename = "test7.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @if_else_multiple_redundant_exprs(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp sgt i32 %0, 2
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = mul nsw i32 %0, %0
  %6 = add nsw i32 %5, %0
  %7 = add nsw i32 %6, 5
  %8 = add nsw i32 %7, 5
  br label %14

9:                                                ; preds = %2
  %10 = mul nsw i32 %0, %0
  %11 = add nsw i32 %10, %0
  %12 = add nsw i32 %11, 3
  %13 = mul nsw i32 %0, %0
  br label %14

14:                                               ; preds = %9, %4
  %.01 = phi i32 [ %5, %4 ], [ %13, %9 ]
  %.0 = phi i32 [ %8, %4 ], [ %12, %9 ]
  %15 = add nsw i32 %.01, %.0
  ret i32 %15
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
