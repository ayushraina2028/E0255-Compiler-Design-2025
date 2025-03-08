; ModuleID = 'test5.c'
source_filename = "test5.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @switch_case(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  switch i32 %6, label %21 [
    i32 0, label %7
    i32 1, label %7
    i32 2, label %14
    i32 3, label %14
  ]

7:                                                ; preds = %2, %2
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %3, align 4
  %10 = mul nsw i32 %8, %9
  %11 = load i32, ptr %3, align 4
  %12 = add nsw i32 %10, %11
  %13 = add nsw i32 %12, 3
  store i32 %13, ptr %5, align 4
  br label %28

14:                                               ; preds = %2, %2
  %15 = load i32, ptr %3, align 4
  %16 = load i32, ptr %3, align 4
  %17 = mul nsw i32 %15, %16
  %18 = load i32, ptr %3, align 4
  %19 = add nsw i32 %17, %18
  %20 = add nsw i32 %19, 3
  store i32 %20, ptr %5, align 4
  br label %28

21:                                               ; preds = %2
  %22 = load i32, ptr %3, align 4
  %23 = load i32, ptr %3, align 4
  %24 = mul nsw i32 %22, %23
  %25 = load i32, ptr %3, align 4
  %26 = add nsw i32 %24, %25
  %27 = add nsw i32 %26, 3
  store i32 %27, ptr %5, align 4
  br label %28

28:                                               ; preds = %21, %14, %7
  %29 = load i32, ptr %5, align 4
  ret i32 %29
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
