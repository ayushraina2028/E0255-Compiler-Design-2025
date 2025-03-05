#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include <bits/stdc++.h>

using namespace std;
using namespace llvm;

namespace {
struct HoistAnticipatedExpressions : public PassInfoMixin<HoistAnticipatedExpressions> {
    
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        

        return PreservedAnalyses::none();
    }
};
}

// Pass registration
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        .APIVersion = LLVM_PLUGIN_API_VERSION,
        .PluginName = "HoistAnticipatedExpressions",
        .PluginVersion = "v0.1",
        .RegisterPassBuilderCallbacks = [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "hoist-anticipated-expressions") {
                        FPM.addPass(HoistAnticipatedExpressions());
                        return true;
                    }
                    return false;
                });
        }
    };
}
