; ModuleID = 'test9.c'
source_filename = "test9.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @not_anticipated_switch(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  switch i32 %6, label %17 [
    i32 0, label %7
    i32 1, label %7
    i32 2, label %7
  ]

7:                                                ; preds = %2, %2, %2
  %8 = load i32, ptr %3, align 4
  %9 = srem i32 %8, 2
  %10 = load i32, ptr %3, align 4
  %11 = srem i32 %10, 2
  %12 = mul nsw i32 %9, %11
  %13 = load i32, ptr %3, align 4
  %14 = srem i32 %13, 2
  %15 = add nsw i32 %12, %14
  %16 = add nsw i32 %15, 3
  store i32 %16, ptr %5, align 4
  br label %24

17:                                               ; preds = %2
  %18 = load i32, ptr %3, align 4
  %19 = load i32, ptr %3, align 4
  %20 = mul nsw i32 %18, %19
  %21 = load i32, ptr %3, align 4
  %22 = add nsw i32 %20, %21
  %23 = add nsw i32 %22, 2
  store i32 %23, ptr %5, align 4
  br label %24

24:                                               ; preds = %17, %7
  %25 = load i32, ptr %5, align 4
  ret i32 %25
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
