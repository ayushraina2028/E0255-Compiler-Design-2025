; ModuleID = 'test2.ll'
source_filename = "test2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @simple_if_else_multiple(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp sgt i32 %0, 3
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = mul nsw i32 %0, %0
  %6 = add nsw i32 %5, %0
  %7 = add nsw i32 %6, 5
  br label %19

8:                                                ; preds = %2
  %9 = icmp sgt i32 %0, 2
  br i1 %9, label %10, label %14

10:                                               ; preds = %8
  %11 = mul nsw i32 %0, %0
  %12 = add nsw i32 %11, %0
  %13 = add nsw i32 %12, 5
  br label %18

14:                                               ; preds = %8
  %15 = mul nsw i32 %0, %0
  %16 = add nsw i32 %15, %0
  %17 = add nsw i32 %16, 5
  br label %18

18:                                               ; preds = %14, %10
  %.1 = phi i32 [ %13, %10 ], [ %17, %14 ]
  br label %19

19:                                               ; preds = %18, %4
  %.0 = phi i32 [ %7, %4 ], [ %.1, %18 ]
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
