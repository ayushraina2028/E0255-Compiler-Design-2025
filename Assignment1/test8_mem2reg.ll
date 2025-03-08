; ModuleID = 'test8.ll'
source_filename = "test8.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @not_anticipated_for_loop(i32 noundef %0, ptr noundef %1) #0 {
  br label %3

3:                                                ; preds = %7, %2
  %.0 = phi i32 [ 0, %2 ], [ %6, %7 ]
  %4 = mul nsw i32 %0, %0
  %5 = srem i32 %4, %0
  %6 = add nsw i32 %.0, 1
  br label %7

7:                                                ; preds = %3
  %8 = icmp slt i32 %6, 10
  br i1 %8, label %3, label %9, !llvm.loop !6

9:                                                ; preds = %7
  ret i32 %5
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
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
