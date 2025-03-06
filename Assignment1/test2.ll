; ModuleID = 'test2.c'
source_filename = "test2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @simple_if_else_multiple(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  %7 = icmp sgt i32 %6, 3
  br i1 %7, label %8, label %15

8:                                                ; preds = %2
  %9 = load i32, ptr %3, align 4
  %10 = load i32, ptr %3, align 4
  %11 = mul nsw i32 %9, %10
  %12 = load i32, ptr %3, align 4
  %13 = add nsw i32 %11, %12
  %14 = add nsw i32 %13, 5
  store i32 %14, ptr %5, align 4
  br label %33

15:                                               ; preds = %2
  %16 = load i32, ptr %3, align 4
  %17 = icmp sgt i32 %16, 2
  br i1 %17, label %18, label %25

18:                                               ; preds = %15
  %19 = load i32, ptr %3, align 4
  %20 = load i32, ptr %3, align 4
  %21 = mul nsw i32 %19, %20
  %22 = load i32, ptr %3, align 4
  %23 = add nsw i32 %21, %22
  %24 = add nsw i32 %23, 5
  store i32 %24, ptr %5, align 4
  br label %32

25:                                               ; preds = %15
  %26 = load i32, ptr %3, align 4
  %27 = load i32, ptr %3, align 4
  %28 = mul nsw i32 %26, %27
  %29 = load i32, ptr %3, align 4
  %30 = add nsw i32 %28, %29
  %31 = add nsw i32 %30, 5
  store i32 %31, ptr %5, align 4
  br label %32

32:                                               ; preds = %25, %18
  br label %33

33:                                               ; preds = %32, %8
  %34 = load i32, ptr %5, align 4
  ret i32 %34
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
