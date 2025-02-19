; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @add(i32 noundef %0, i32 noundef %1) #0 !dbg !10 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, ptr %4, align 4
    #dbg_declare(ptr %4, !15, !DIExpression(), !16)
  store i32 %1, ptr %5, align 4
    #dbg_declare(ptr %5, !17, !DIExpression(), !18)
  %6 = load i32, ptr %4, align 4, !dbg !19
  %7 = load i32, ptr %5, align 4, !dbg !21
  %8 = icmp sgt i32 %6, %7, !dbg !22
  br i1 %8, label %9, label %13, !dbg !22

9:                                                ; preds = %2
  %10 = load i32, ptr %4, align 4, !dbg !23
  %11 = load i32, ptr %5, align 4, !dbg !25
  %12 = add nsw i32 %10, %11, !dbg !26
  store i32 %12, ptr %3, align 4, !dbg !27
  br label %17, !dbg !27

13:                                               ; preds = %2
  %14 = load i32, ptr %5, align 4, !dbg !28
  %15 = load i32, ptr %4, align 4, !dbg !30
  %16 = add nsw i32 %14, %15, !dbg !31
  store i32 %16, ptr %3, align 4, !dbg !32
  br label %17, !dbg !32

17:                                               ; preds = %13, %9
  %18 = load i32, ptr %3, align 4, !dbg !33
  ret i32 %18, !dbg !33
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !34 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, ptr %1, align 4
    #dbg_declare(ptr %2, !37, !DIExpression(), !38)
  store i32 5, ptr %2, align 4, !dbg !38
    #dbg_declare(ptr %3, !39, !DIExpression(), !40)
  store i32 10, ptr %3, align 4, !dbg !40
    #dbg_declare(ptr %4, !41, !DIExpression(), !42)
  %5 = load i32, ptr %2, align 4, !dbg !43
  %6 = load i32, ptr %3, align 4, !dbg !44
  %7 = call i32 @add(i32 noundef %5, i32 noundef %6), !dbg !45
  store i32 %7, ptr %4, align 4, !dbg !42
  %8 = load i32, ptr %4, align 4, !dbg !46
  ret i32 %8, !dbg !47
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (https://github.com/llvm/llvm-project.git 6a3007683bf2fa05989c12c787f5547788d09178)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/ayushraina/Desktop/CompilerDesign/llvm-pass-demo/MyFirstPass", checksumkind: CSK_MD5, checksum: "3067804d87baa89ebbf89deb74047872")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 6a3007683bf2fa05989c12c787f5547788d09178)"}
!10 = distinct !DISubprogram(name: "add", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 1, column: 13, scope: !10)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !1, line: 1, type: !13)
!18 = !DILocation(line: 1, column: 20, scope: !10)
!19 = !DILocation(line: 2, column: 9, scope: !20)
!20 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 9)
!21 = !DILocation(line: 2, column: 13, scope: !20)
!22 = !DILocation(line: 2, column: 11, scope: !20)
!23 = !DILocation(line: 3, column: 16, scope: !24)
!24 = distinct !DILexicalBlock(scope: !20, file: !1, line: 2, column: 16)
!25 = !DILocation(line: 3, column: 20, scope: !24)
!26 = !DILocation(line: 3, column: 18, scope: !24)
!27 = !DILocation(line: 3, column: 9, scope: !24)
!28 = !DILocation(line: 5, column: 16, scope: !29)
!29 = distinct !DILexicalBlock(scope: !20, file: !1, line: 4, column: 12)
!30 = !DILocation(line: 5, column: 20, scope: !29)
!31 = !DILocation(line: 5, column: 18, scope: !29)
!32 = !DILocation(line: 5, column: 9, scope: !29)
!33 = !DILocation(line: 7, column: 1, scope: !10)
!34 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 9, type: !35, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!35 = !DISubroutineType(types: !36)
!36 = !{!13}
!37 = !DILocalVariable(name: "x", scope: !34, file: !1, line: 10, type: !13)
!38 = !DILocation(line: 10, column: 9, scope: !34)
!39 = !DILocalVariable(name: "y", scope: !34, file: !1, line: 10, type: !13)
!40 = !DILocation(line: 10, column: 16, scope: !34)
!41 = !DILocalVariable(name: "result", scope: !34, file: !1, line: 11, type: !13)
!42 = !DILocation(line: 11, column: 9, scope: !34)
!43 = !DILocation(line: 11, column: 22, scope: !34)
!44 = !DILocation(line: 11, column: 25, scope: !34)
!45 = !DILocation(line: 11, column: 18, scope: !34)
!46 = !DILocation(line: 12, column: 12, scope: !34)
!47 = !DILocation(line: 12, column: 5, scope: !34)
