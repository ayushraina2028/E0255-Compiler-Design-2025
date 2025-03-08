; ModuleID = 'test7.c'
source_filename = "test7.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @if_else_multiple_redundant_exprs(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %7 = load i32, ptr %3, align 4
  %8 = icmp sgt i32 %7, 2
  br i1 %8, label %9, label %20

9:                                                ; preds = %2
  %10 = load i32, ptr %3, align 4
  %11 = load i32, ptr %3, align 4
  %12 = mul nsw i32 %10, %11
  store i32 %12, ptr %5, align 4
  %13 = load i32, ptr %5, align 4
  %14 = load i32, ptr %3, align 4
  %15 = add nsw i32 %13, %14
  store i32 %15, ptr %6, align 4
  %16 = load i32, ptr %6, align 4
  %17 = add nsw i32 %16, 5
  store i32 %17, ptr %6, align 4
  %18 = load i32, ptr %6, align 4
  %19 = add nsw i32 %18, 5
  store i32 %19, ptr %6, align 4
  br label %32

20:                                               ; preds = %2
  %21 = load i32, ptr %3, align 4
  %22 = load i32, ptr %3, align 4
  %23 = mul nsw i32 %21, %22
  store i32 %23, ptr %5, align 4
  %24 = load i32, ptr %5, align 4
  %25 = load i32, ptr %3, align 4
  %26 = add nsw i32 %24, %25
  store i32 %26, ptr %6, align 4
  %27 = load i32, ptr %6, align 4
  %28 = add nsw i32 %27, 3
  store i32 %28, ptr %6, align 4
  %29 = load i32, ptr %3, align 4
  %30 = load i32, ptr %3, align 4
  %31 = mul nsw i32 %29, %30
  store i32 %31, ptr %5, align 4
  br label %32

32:                                               ; preds = %20, %9
  %33 = load i32, ptr %5, align 4
  %34 = load i32, ptr %6, align 4
  %35 = add nsw i32 %33, %34
  ret i32 %35
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
