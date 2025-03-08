; Function Attrs: noinline nounwind uwtable
define dso_local i32 @if_else_math_call(i32 noundef %0, ptr noundef %1) #0 {
  %3 = icmp sgt i32 %0, 3
  %hoisted1 = mul nsw i32 %0, %0
  %hoisted2 = sitofp i32 %0 to double
  %hoisted3 = sitofp i32 %hoisted1 to double
  %hoisted4 = call double @exp(double %hoisted2)
  %hoisted5 = fadd nnan double %hoisted3, %hoisted4
  br i1 %3, label %4, label %6

4:                                                ; preds = %2
  %5 = fadd double %hoisted5, 5.000000e+00
  br label %13

6:                                                ; preds = %2
  %7 = icmp sgt i32 %0, 2
  br i1 %7, label %8, label %10

8:                                                ; preds = %6
  %9 = fadd double %hoisted5, 3.000000e+00
  br label %12

10:                                               ; preds = %6
  %11 = fadd double %hoisted5, 1.000000e+00
  br label %12

12:                                               ; preds = %10, %8
  %.1 = phi double [ %9, %8 ], [ %11, %10 ]
  br label %13

13:                                               ; preds = %12, %4
  %.0 = phi double [ %5, %4 ], [ %.1, %12 ]
  %14 = fptosi double %.0 to i32
  ret i32 %14
}
