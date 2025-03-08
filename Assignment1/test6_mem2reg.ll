; if/else with a math function call.

attributes #3 = { nounwind }

; Function Attrs: nounwind
declare dso_local double @exp(double noundef) #2

; Function Attrs: nounwind uwtable
; CHECK: @if_else_math_call
define dso_local i32 @if_else_math_call(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp ugt i32 %0, 3
  br i1 %3, label %4, label %12

  ; CHECK: mul
  ; CHECK: call double @exp
  ; CHECK-NOT: call

4:                                                ; preds = %2
  %5 = mul i32 %0, %0
  %6 = uitofp i32 %5 to double
  %7 = uitofp i32 %0 to double
  %8 = call double @exp(double noundef %7) #3
  %9 = fadd double %6, %8
  %10 = fadd double %9, 5.000000e+00
  %11 = fptoui double %10 to i32
  br label %30

12:                                               ; preds = %2
  %13 = icmp ugt i32 %0, 2
  br i1 %13, label %14, label %22

14:                                               ; preds = %12
  %15 = mul i32 %0, %0
  %16 = uitofp i32 %15 to double
  %17 = uitofp i32 %0 to double
  %18 = call double @exp(double noundef %17) #3
  %19 = fadd double %16, %18
  %20 = fadd double %19, 3.000000e+00
  %21 = fptoui double %20 to i32
  br label %30

22:                                               ; preds = %12
  %23 = mul i32 %0, %0
  %24 = uitofp i32 %23 to double
  %25 = uitofp i32 %0 to double
  %26 = call double @exp(double noundef %25) #3
  %27 = fadd double %24, %26
  %28 = fadd double %27, 1.000000e+00
  %29 = fptoui double %28 to i32
  br label %30

30:                                               ; preds = %14, %22, %4
  %31 = phi i32 [ %11, %4 ], [ %21, %14 ], [ %29, %22 ]
  ret i32 %31
}
