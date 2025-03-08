; Function Attrs: noinline nounwind uwtable
define dso_local i32 @if_else_math_call(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp sgt i32 %0, 3
  %hoisted1 = mul nsw i32 %0, %0
  br i1 %3, label %4, label %10

4:                                                ; preds = %2
  %5 = sitofp i32 %hoisted1 to double
  %6 = sitofp i32 %0 to double
  %7 = call double @exp(double noundef %6) #2
  %8 = fadd double %5, %7
  %9 = fadd double %8, 5.000000e+00
  br label %25

10:                                               ; preds = %2
  %11 = icmp sgt i32 %0, 2
  br i1 %11, label %12, label %18

12:                                               ; preds = %10
  %13 = sitofp i32 %hoisted1 to double
  %14 = sitofp i32 %0 to double
  %15 = call double @exp(double noundef %14) #2
  %16 = fadd double %13, %15
  %17 = fadd double %16, 3.000000e+00
  br label %24

18:                                               ; preds = %10
  %19 = sitofp i32 %hoisted1 to double
  %20 = sitofp i32 %0 to double
  %21 = call double @exp(double noundef %20) #2
  %22 = fadd double %19, %21
  %23 = fadd double %22, 1.000000e+00
  br label %24

24:                                               ; preds = %18, %12
  %.1 = phi double [ %17, %12 ], [ %23, %18 ]
  br label %25

25:                                               ; preds = %24, %4
  %.0 = phi double [ %9, %4 ], [ %.1, %24 ]
  %26 = fptosi double %.0 to i32
  ret i32 %26
}
