; ModuleID = 'test6.ll'
source_filename = "test6.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @if_else_math_call(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp sgt i32 %0, 3
  br i1 %3, label %4, label %11

4:                                                ; preds = %2
  %5 = mul nsw i32 %0, %0
  %6 = sitofp i32 %5 to double
  %7 = sitofp i32 %0 to double
  %8 = call double @exp(double noundef %7) #2
  %9 = fadd double %6, %8
  %10 = fadd double %9, 5.000000e+00
  br label %28

11:                                               ; preds = %2
  %12 = icmp sgt i32 %0, 2
  br i1 %12, label %13, label %20

13:                                               ; preds = %11
  %14 = mul nsw i32 %0, %0
  %15 = sitofp i32 %14 to double
  %16 = sitofp i32 %0 to double
  %17 = call double @exp(double noundef %16) #2
  %18 = fadd double %15, %17
  %19 = fadd double %18, 3.000000e+00
  br label %27

20:                                               ; preds = %11
  %21 = mul nsw i32 %0, %0
  %22 = sitofp i32 %21 to double
  %23 = sitofp i32 %0 to double
  %24 = call double @exp(double noundef %23) #2
  %25 = fadd double %22, %24
  %26 = fadd double %25, 1.000000e+00
  br label %27

27:                                               ; preds = %20, %13
  %.1 = phi double [ %19, %13 ], [ %26, %20 ]
  br label %28

28:                                               ; preds = %27, %4
  %.0 = phi double [ %10, %4 ], [ %.1, %27 ]
  %29 = fptosi double %.0 to i32
  ret i32 %29
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
