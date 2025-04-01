#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/PassManager.h"
#include "MyPass/MyPass.h"

using namespace mlir;

namespace {
struct ModifyLoopBoundsPass : public PassWrapper<ModifyLoopBoundsPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([](affine::AffineForOp loop) {
      OpBuilder builder(loop);

      // Modify upper bound if it's constant
      if (loop.hasConstantUpperBound()) {
        int64_t newUpperBound = loop.getConstantUpperBound() - 1;
        loop.setConstantUpperBound(newUpperBound);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createModifyLoopBoundsPass() {
  return std::make_unique<ModifyLoopBoundsPass>();
}

// Register pass
namespace {
#define GEN_PASS_REGISTRATION
#include "MyPass/MyPass.h.inc"
} // namespace
