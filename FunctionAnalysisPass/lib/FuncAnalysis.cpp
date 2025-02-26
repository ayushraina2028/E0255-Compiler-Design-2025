#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct FunctionAnalysisPass : public PassInfoMixin<FunctionAnalysisPass> {

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Function: " << F.getName() << "\n";
        errs() << "Number of instructions: " << F.getInstructionCount() << "\n";
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
        LLVM_PLUGIN_API_VERSION, "FunctionAnalysisPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "function-analysis") {
                        FPM.addPass(FunctionAnalysisPass());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "Function Analysis Pass plugin loaded!\n";
        }
    };
}

/* opt -load-pass-plugin=./build/libFunctionAnalysisPass.so -passes="function-analysis" test.ll -o /dev/null   (to run the pass) */
