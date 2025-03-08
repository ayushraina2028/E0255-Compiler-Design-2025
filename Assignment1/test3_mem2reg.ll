; No optimization needs to be performed on the memory form (it's okay to not
; optimize and so we don't check anything here except that the pass shouldn't
; crash on this.
; CHECK: if_else_memory
define dso_local i32 @if_else_memory(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store i32 0, ptr %3, align 4
  store i32 %0, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  %7 = load i32, ptr %4, align 4
  store i32 %7, ptr %6, align 4
  %8 = load i32, ptr %6, align 4
  %9 = icmp ugt i32 %8, 2
  br i1 %9, label %10, label %17

10:                                               ; preds = %2
  %11 = load i32, ptr %6, align 4
  %12 = load i32, ptr %6, align 4
  %13 = mul i32 %11, %12
  %14 = load i32, ptr %6, align 4
  %15 = add i32 %13, %14
  %16 = add i32 %15, 5
  store i32 %16, ptr %6, align 4
  br label %24

17:                                               ; preds = %2
  %18 = load i32, ptr %6, align 4
  %19 = load i32, ptr %6, align 4
  %20 = mul i32 %18, %19
  %21 = load i32, ptr %6, align 4
  %22 = add i32 %20, %21
  %23 = add i32 %22, 3
  store i32 %23, ptr %6, align 4
  br label %24

24:                                               ; preds = %17, %10
  %25 = load i32, ptr %6, align 4
  ret i32 %25
}