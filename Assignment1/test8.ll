; ModuleID = 'test8.c'
source_filename = "test8.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @not_anticipated_for_loop(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  store i32 0, ptr %6, align 4
  br label %7

7:                                                ; preds = %15, %2
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %3, align 4
  %10 = mul nsw i32 %8, %9
  %11 = load i32, ptr %3, align 4
  %12 = srem i32 %10, %11
  store i32 %12, ptr %5, align 4
  %13 = load i32, ptr %6, align 4
  %14 = add nsw i32 %13, 1
  store i32 %14, ptr %6, align 4
  br label %15

15:                                               ; preds = %7
  %16 = load i32, ptr %6, align 4
  %17 = icmp slt i32 %16, 10
  br i1 %17, label %7, label %18, !llvm.loop !6

18:                                               ; preds = %15
  %19 = load i32, ptr %5, align 4
  ret i32 %19
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
