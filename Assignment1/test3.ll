; ModuleID = 'test3.c'
source_filename = "test3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @test_complex() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca [10 x [10 x i32]], align 16
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store i32 10, ptr %1, align 4
  store i32 20, ptr %2, align 4
  store i32 30, ptr %3, align 4
  %12 = load i32, ptr %1, align 4
  %13 = load i32, ptr %2, align 4
  %14 = icmp sgt i32 %12, %13
  br i1 %14, label %15, label %28

15:                                               ; preds = %0
  %16 = load i32, ptr %2, align 4
  %17 = load i32, ptr %3, align 4
  %18 = icmp sgt i32 %16, %17
  br i1 %18, label %19, label %23

19:                                               ; preds = %15
  %20 = load i32, ptr %2, align 4
  %21 = load i32, ptr %3, align 4
  %22 = add nsw i32 %20, %21
  store i32 %22, ptr %1, align 4
  br label %27

23:                                               ; preds = %15
  %24 = load i32, ptr %1, align 4
  %25 = load i32, ptr %3, align 4
  %26 = add nsw i32 %24, %25
  store i32 %26, ptr %2, align 4
  br label %27

27:                                               ; preds = %23, %19
  br label %32

28:                                               ; preds = %0
  %29 = load i32, ptr %1, align 4
  %30 = load i32, ptr %2, align 4
  %31 = add nsw i32 %29, %30
  store i32 %31, ptr %3, align 4
  br label %32

32:                                               ; preds = %28, %27
  store i32 0, ptr %5, align 4
  br label %33

33:                                               ; preds = %57, %32
  %34 = load i32, ptr %5, align 4
  %35 = icmp slt i32 %34, 10
  br i1 %35, label %36, label %60

36:                                               ; preds = %33
  %37 = load i32, ptr %1, align 4
  %38 = load i32, ptr %2, align 4
  %39 = mul nsw i32 %37, %38
  store i32 %39, ptr %6, align 4
  store i32 0, ptr %7, align 4
  br label %40

40:                                               ; preds = %53, %36
  %41 = load i32, ptr %7, align 4
  %42 = icmp slt i32 %41, 10
  br i1 %42, label %43, label %56

43:                                               ; preds = %40
  %44 = load i32, ptr %6, align 4
  %45 = load i32, ptr %7, align 4
  %46 = add nsw i32 %44, %45
  %47 = load i32, ptr %5, align 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds [10 x [10 x i32]], ptr %4, i64 0, i64 %48
  %50 = load i32, ptr %7, align 4
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds [10 x i32], ptr %49, i64 0, i64 %51
  store i32 %46, ptr %52, align 4
  br label %53

53:                                               ; preds = %43
  %54 = load i32, ptr %7, align 4
  %55 = add nsw i32 %54, 1
  store i32 %55, ptr %7, align 4
  br label %40, !llvm.loop !6

56:                                               ; preds = %40
  br label %57

57:                                               ; preds = %56
  %58 = load i32, ptr %5, align 4
  %59 = add nsw i32 %58, 1
  store i32 %59, ptr %5, align 4
  br label %33, !llvm.loop !8

60:                                               ; preds = %33
  %61 = load i32, ptr %1, align 4
  %62 = load i32, ptr %2, align 4
  %63 = add nsw i32 %61, %62
  store i32 %63, ptr %8, align 4
  %64 = load i32, ptr %2, align 4
  %65 = load i32, ptr %3, align 4
  %66 = add nsw i32 %64, %65
  store i32 %66, ptr %9, align 4
  %67 = load i32, ptr %1, align 4
  %68 = load i32, ptr %2, align 4
  %69 = add nsw i32 %67, %68
  store i32 %69, ptr %10, align 4
  %70 = load i32, ptr %2, align 4
  %71 = load i32, ptr %3, align 4
  %72 = add nsw i32 %70, %71
  store i32 %72, ptr %11, align 4
  %73 = load i32, ptr %8, align 4
  %74 = load i32, ptr %9, align 4
  %75 = add nsw i32 %73, %74
  %76 = load i32, ptr %10, align 4
  %77 = add nsw i32 %75, %76
  %78 = load i32, ptr %11, align 4
  %79 = add nsw i32 %77, %78
  ret i32 %79
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
!8 = distinct !{!8, !7}
