#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct InstructionLevelAnalysis : public PassInfoMixin<InstructionLevelAnalysis> {

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Function: " << F.getName() << "\n";
        for (auto &BB : F) {
            for (auto &I : BB) {
                errs() << "Instruction: " << I.getOpcodeName() << "\n";
            }
        }
        return PreservedAnalyses::all();
    }

    static bool isRequired() {
        return true;
    }
};  

}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "InstructionLevelAnalysis", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "instruction-level-analysis") {
                        FPM.addPass(InstructionLevelAnalysis());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "Instruction Level Analysis Pass plugin loaded!\n";
        }
    };
}

/* opt -load-pass-plugin=./build/libInstructionLevelAnalysis.so -passes="instruction-level-analysis" test.ll -o -disable-output   (to run the pass) */
