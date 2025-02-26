#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"   
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/ADT/DenseMap.h"
#include <bits/stdc++.h>

using namespace llvm;
using namespace std;

namespace {

struct AnticipatedExpressions : public PassInfoMixin<AnticipatedExpressions> {

    using ExpressionSet = set<Value *>;
    DenseMap<BasicBlock *, ExpressionSet> IN;
    DenseMap<BasicBlock *, ExpressionSet> OUT;

    DenseMap<BasicBlock *, ExpressionSet> V_USE;
    DenseMap<BasicBlock *, ExpressionSet> V_DEF;

    ExpressionSet UniversalSet;

    // Defining structure to store the expression
    struct Expression {
        Value* LHS;
        Value* RHS;
        unsigned Opcode;

        // Constructor
        Expression(Value* LHS, Value* RHS, unsigned Opcode) {
            this->LHS = LHS;
            this->RHS = RHS;
            this->Opcode = Opcode;
        }

        // Function for Debugging
        void dump() const {
            errs() << "Left Operand: " << *LHS << "\n";
            errs() << "Right Operand: " << *RHS << "\n";
        }
    };

    // Function to check if the value is defined in a instruction or not
    bool isDefinedInInstruction(Value* V, Instruction& I) {
        Value* defined = dyn_cast<Value>(&I);
        if(defined != nullptr and defined == V) {
            return true;
        }
        return false;
    }

    // Function to check if the instruction is a binary operation or not
    bool isBinaryOperation(Instruction &I) {

        if(!I.isBinaryOp()) {
            return false;
        }

        errs() << "Found Binary Operation: " << I.getOpcodeName() << "\n";
        return true;
    }

    // Function to create expressions from instructions
    Expression* CreateExpressionFromInstruction(Instruction& I) {
        if(!isBinaryOperation(I)) {
            return nullptr;
        }

        Value* LHS = dyn_cast<BinaryOperator>(&I)->getOperand(0);
        Value* RHS = dyn_cast<BinaryOperator>(&I)->getOperand(1);
        unsigned Opcode = I.getOpcode();

        return new Expression(LHS, RHS, Opcode);
    }

    // Function to Build the Universal Set
    void buildUniversalSet(Function &F) {
        errs() << "Building Universal Set\n";

        for(BasicBlock &BB : F) {
            for(Instruction &I : BB) {

                Expression* Expr = CreateExpressionFromInstruction(I);
                if(Expr) {
                    UniversalSet.insert(&I);
                    //Expr->dump(); // Debugging
                    delete Expr;
                }

            }
        }
        errs() << "Universal Set Size: " << UniversalSet.size() << "\n";
    }

    bool isDefinedIn(Value* V, Instruction &I) {

        Expression* Expr = CreateExpressionFromInstruction(I);
        if(Expr == nullptr) return false;

        bool ans = (Expr->LHS == V or Expr->RHS == V);
        
        delete Expr;
        return ans;
    }

    bool isDefinedBefore(Value* V, Instruction &I, BasicBlock &BB) {
        // Check if the value is defined in the same instruction

        for(Instruction &PrevInst : BB) { // ordered traversal

            if(&PrevInst == &I) return false;
            if(isDefinedIn(V,PrevInst)) return true;

        }

        return false;
    }

    // Function to Analyze expressions in a Basic Block
    void AnalyzeExpressionsInBasicBlock(BasicBlock &BB) {
        errs() << "Analyzing Basic Block for Initializing V_USE and V_DEF\n";

        for(Instruction &I : BB) {
            Expression* Expr = CreateExpressionFromInstruction(I);
            if(Expr == nullptr) continue;

            // Check if the operands are defined before this instruction
            bool ifLeftOperandDefined = isDefinedBefore(Expr->LHS,I,BB);
            bool ifRightOperandDefined = isDefinedBefore(Expr->RHS,I,BB);

            if(!ifLeftOperandDefined and !ifRightOperandDefined) {
                V_USE[&BB].insert(&I);
                errs() << "Added to V_USE: " << I << "\n";
                // Expr->dump();
            }
            else {
                V_DEF[&BB].insert(&I);
                errs() << "Added to V_DEF: " << I << "\n";
                // Expr->dump();
            }

            if(ifLeftOperandDefined) errs() << "Left Operand Defined\n";    
            else errs() << "Left Operand Not Defined\n";

            if(ifRightOperandDefined) errs() << "Right Operand Defined\n";    
            else errs() << "Right Operand Not Defined\n";

            delete Expr;
        }
    }

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Function: " << F.getName() << "\n";
        buildUniversalSet(F);

        // First Step is to iterate through all basic blocks and instructions to identify the expressions of form (B op C) in the code
        for(BasicBlock &BB : F) {
            errs() << "Basic Block \n";

            // Initialize IN sets for all basic blocks
            IN[&BB] = UniversalSet;
            
            // Remaining Initializations
            OUT[&BB] = ExpressionSet();
            V_USE[&BB] = ExpressionSet();
            V_DEF[&BB] = ExpressionSet();

            errs() << "IN Set Size: " << IN[&BB].size() << "\n";
            // errs() << "OUT Set Size: " << OUT[&BB].size() << "\n"; (debugging)
            // errs() << "V_USE Set Size: " << V_USE[&BB].size() << "\n"; (debugging)
            // errs() << "V_DEF Set Size: " << V_DEF[&BB].size() << "\n"; (debugging)
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
        LLVM_PLUGIN_API_VERSION, "AnticipatedExpressions", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register the pass
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "anticipated-expressions") {
                        FPM.addPass(AnticipatedExpressions());
                        return true;
                    }
                    return false;
                });
            // Add debug output to verify registration
            errs() << "Anticipated Expression plugin loaded!\n";
        }
    };
}

/* opt -load-pass-plugin=./build/libAnticipatedExpressions.so -passes="anticipated-expressions test.ll -o -disable-output   (to run the pass) */
