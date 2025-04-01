add_mlir_dialect_library(MLIRLoopBoundPass
  MyPass.cpp

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)

mlir_check_all_link_libraries(MLIRLoopBoundPass)
