; ModuleID = 'test6.c'
source_filename = "test6.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @if_else_math_call(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca double, align 8
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %6 = load i32, ptr %3, align 4
  %7 = icmp sgt i32 %6, 3
  br i1 %7, label %8, label %18

8:                                                ; preds = %2
  %9 = load i32, ptr %3, align 4
  %10 = load i32, ptr %3, align 4
  %11 = mul nsw i32 %9, %10
  %12 = sitofp i32 %11 to double
  %13 = load i32, ptr %3, align 4
  %14 = sitofp i32 %13 to double
  %15 = call double @exp(double noundef %14) #2
  %16 = fadd double %12, %15
  %17 = fadd double %16, 5.000000e+00
  store double %17, ptr %5, align 8
  br label %42

18:                                               ; preds = %2
  %19 = load i32, ptr %3, align 4
  %20 = icmp sgt i32 %19, 2
  br i1 %20, label %21, label %31

21:                                               ; preds = %18
  %22 = load i32, ptr %3, align 4
  %23 = load i32, ptr %3, align 4
  %24 = mul nsw i32 %22, %23
  %25 = sitofp i32 %24 to double
  %26 = load i32, ptr %3, align 4
  %27 = sitofp i32 %26 to double
  %28 = call double @exp(double noundef %27) #2
  %29 = fadd double %25, %28
  %30 = fadd double %29, 3.000000e+00
  store double %30, ptr %5, align 8
  br label %41

31:                                               ; preds = %18
  %32 = load i32, ptr %3, align 4
  %33 = load i32, ptr %3, align 4
  %34 = mul nsw i32 %32, %33
  %35 = sitofp i32 %34 to double
  %36 = load i32, ptr %3, align 4
  %37 = sitofp i32 %36 to double
  %38 = call double @exp(double noundef %37) #2
  %39 = fadd double %35, %38
  %40 = fadd double %39, 1.000000e+00
  store double %40, ptr %5, align 8
  br label %41

41:                                               ; preds = %31, %21
  br label %42

42:                                               ; preds = %41, %8
  %43 = load double, ptr %5, align 8
  %44 = fptosi double %43 to i32
  ret i32 %44
}

; Function Attrs: nounwind
declare double @exp(double noundef) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 6a3007683bf2fa05989c12c787f5547788d09178)"}
