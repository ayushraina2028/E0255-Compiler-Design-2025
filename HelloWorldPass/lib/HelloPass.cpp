#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct HelloWorldPass : public PassInfoMixin<HelloWorldPass> {

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << F.getName() << '\n';
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
        LLVM_PLUGIN_API_VERSION, "HelloWorldPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "hello-world") {
                        FPM.addPass(HelloWorldPass());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "Hello World Pass plugin loaded!\n";
        }
    };
}

/* opt -load-pass-plugin=./build/libHelloWorldPass.so -passes="hello-world" test.ll -o /dev/null   (to run the pass) */
