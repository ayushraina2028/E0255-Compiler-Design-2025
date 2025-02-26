#include "llvm/IR/PassManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct ConstantFolding : public PassInfoMixin<ConstantFolding> {

     PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        bool Changed = false;

        for (auto &BB : F) {
            for (auto It = BB.begin(); It != BB.end(); ) {
                Instruction *Inst = &*It++;

                // Propagate constant loads
                if (auto *Load = dyn_cast<LoadInst>(Inst)) {
                    if (auto *Alloca = dyn_cast<AllocaInst>(Load->getPointerOperand())) {
                        for (auto UI = Alloca->user_begin(); UI != Alloca->user_end(); ++UI) {
                            if (auto *Store = dyn_cast<StoreInst>(*UI)) {
                                if (auto *ConstVal = dyn_cast<ConstantInt>(Store->getValueOperand())) {
                                    Load->replaceAllUsesWith(ConstVal);
                                    Changed = true;
                                }
                            }
                        }
                    }
                }

                // Fold binary operations with constants
                if (auto *BinOp = dyn_cast<BinaryOperator>(Inst)) {
                    if (auto *C1 = dyn_cast<ConstantInt>(BinOp->getOperand(0))) {
                        if (auto *C2 = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
                            Constant *Folded = ConstantExpr::get(BinOp->getOpcode(), C1, C2);
                            BinOp->replaceAllUsesWith(Folded);
                            BinOp->eraseFromParent();
                            Changed = true;
                        }
                    }
                }
            }
        }

        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

    static bool isRequired() {
        return true;
    }
};  

}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "ConstantFolding", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "constant-folding") {
                        FPM.addPass(ConstantFolding());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "Constant Folding Pass plugin loaded!\n";
        }
    };
}

/* opt -load-pass-plugin=./build/libConstantFolding.so -passes="constant-folding" test.ll -S -o optimizedIR.ll   (to run the pass) */
