#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct BasicBlockAnalysis : public PassInfoMixin<BasicBlockAnalysis> {

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Function: " << F.getName() << "\n";
        for (auto &BB : F) {
            errs() << "BasicBlock: " << BB.getName() << " has " << BB.size() << " instructions\n";
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
        LLVM_PLUGIN_API_VERSION, "BasicBlockAnalysis", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "basic-block-analysis") {
                        FPM.addPass(BasicBlockAnalysis());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "Basic Block Analysis Pass plugin loaded!\n";
        }
    };
}

/* opt -load-pass-plugin=./build/libBasicBlockAnalysis.so -passes="basic-block-analysis" test.ll -o -disable-output   (to run the pass) */
