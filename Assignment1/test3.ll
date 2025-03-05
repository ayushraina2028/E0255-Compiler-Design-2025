; ModuleID = 'test3.c'
source_filename = "test3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @test_function() #0 {
  %1 = alloca [10 x i32], align 16
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = getelementptr inbounds [10 x i32], ptr %1, i64 0, i64 0
  store ptr %10, ptr %2, align 8
  %11 = load ptr, ptr %2, align 8
  store i32 5, ptr %11, align 4
  %12 = load ptr, ptr %2, align 8
  %13 = load i32, ptr %12, align 4
  store i32 %13, ptr %3, align 4
  store i32 10, ptr %4, align 4
  store i32 20, ptr %5, align 4
  %14 = load i32, ptr %4, align 4
  %15 = load i32, ptr %5, align 4
  %16 = add nsw i32 %14, %15
  store i32 %16, ptr %6, align 4
  %17 = load i32, ptr %4, align 4
  %18 = load i32, ptr %5, align 4
  %19 = add nsw i32 %17, %18
  store i32 %19, ptr %7, align 4
  %20 = load i32, ptr %4, align 4
  %21 = load i32, ptr %5, align 4
  %22 = icmp sgt i32 %20, %21
  br i1 %22, label %23, label %29

23:                                               ; preds = %0
  %24 = load i32, ptr %4, align 4
  %25 = load i32, ptr %5, align 4
  %26 = add nsw i32 %24, %25
  store i32 %26, ptr %6, align 4
  %27 = load i32, ptr %6, align 4
  %28 = mul nsw i32 %27, 2
  store i32 %28, ptr %3, align 4
  br label %35

29:                                               ; preds = %0
  %30 = load i32, ptr %4, align 4
  %31 = load i32, ptr %5, align 4
  %32 = add nsw i32 %30, %31
  store i32 %32, ptr %6, align 4
  %33 = load i32, ptr %6, align 4
  %34 = mul nsw i32 %33, 3
  store i32 %34, ptr %3, align 4
  br label %35

35:                                               ; preds = %29, %23
  %36 = load i32, ptr %3, align 4
  switch i32 %36, label %45 [
    i32 1, label %37
    i32 2, label %41
  ]

37:                                               ; preds = %35
  %38 = load i32, ptr %4, align 4
  %39 = load i32, ptr %5, align 4
  %40 = add nsw i32 %38, %39
  store i32 %40, ptr %6, align 4
  br label %46

41:                                               ; preds = %35
  %42 = load i32, ptr %4, align 4
  %43 = load i32, ptr %5, align 4
  %44 = add nsw i32 %42, %43
  store i32 %44, ptr %7, align 4
  br label %46

45:                                               ; preds = %35
  store i32 0, ptr %3, align 4
  br label %46

46:                                               ; preds = %45, %41, %37
  store i32 42, ptr %8, align 4
  store i32 0, ptr %9, align 4
  br label %47

47:                                               ; preds = %59, %46
  %48 = load i32, ptr %9, align 4
  %49 = icmp slt i32 %48, 10
  br i1 %49, label %50, label %62

50:                                               ; preds = %47
  %51 = load i32, ptr %8, align 4
  %52 = mul nsw i32 %51, 2
  %53 = load i32, ptr %9, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds [10 x i32], ptr %1, i64 0, i64 %54
  store i32 %52, ptr %55, align 4
  %56 = load i32, ptr %9, align 4
  %57 = load i32, ptr %6, align 4
  %58 = add nsw i32 %57, %56
  store i32 %58, ptr %6, align 4
  br label %59

59:                                               ; preds = %50
  %60 = load i32, ptr %9, align 4
  %61 = add nsw i32 %60, 1
  store i32 %61, ptr %9, align 4
  br label %47, !llvm.loop !6

62:                                               ; preds = %47
  %63 = load i32, ptr %6, align 4
  %64 = load i32, ptr %7, align 4
  %65 = add nsw i32 %63, %64
  ret i32 %65
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
