#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct MyFirstPass : public PassInfoMixin<MyFirstPass> {
    PreservedAnalyses run(Function &F,
                                      FunctionAnalysisManager &AM) {
        errs() << F.getName() << "\n";
        return PreservedAnalyses::all();
        }

    // Add this to make the pass print more information
    static bool isRequired() { return true; }
};
}

// This is the core interface for pass plugins
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "MyFirstPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "myfirstpass") {
                        FPM.addPass(MyFirstPass());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "MyFirstPass plugin loaded!\n";
        }
    };
}