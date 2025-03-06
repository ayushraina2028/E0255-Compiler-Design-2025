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
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <bits/stdc++.h>

using namespace std;
using namespace llvm;

namespace {
    struct HoistAnticipatedExpressions : public PassInfoMixin<HoistAnticipatedExpressions> {
        using Expression = tuple<unsigned int, Value*, Value*>;
    
        PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
            DenseMap<Expression, unsigned int> ExpressionToBitMap;
            unsigned int nextBitIndex = 0;

            DenseMap<BasicBlock*, BitVector> IN;
            DenseMap<BasicBlock*, BitVector> OUT;
    
            // Step 1: Collect unique expressions
            for (auto &BB : F) {
                for (auto &Inst : BB) {
                    if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
                        Value *B = BinOp->getOperand(0);    
                        Value *C = BinOp->getOperand(1);
                        unsigned int opcode = BinOp->getOpcode();
    
                        if (BinOp->isCommutative()) {
                            tie(B, C) = minmax(B, C);
                        }
    
                        Expression exprKey = make_tuple(opcode, B, C);
                        if (ExpressionToBitMap.find(exprKey) == ExpressionToBitMap.end()) {
                            ExpressionToBitMap[exprKey] = nextBitIndex++;
                        }
                    }
                }
            }
    
            // Step 2: Initialize Gen and Kill sets for each basic block
            DenseMap<BasicBlock *, BitVector> Gen, Kill;
            for (auto &BB : F) {
                Gen[&BB] = BitVector(nextBitIndex, false);
                Kill[&BB] = BitVector(nextBitIndex, false);
            }
    
            // Step 3: Compute Gen and Kill sets
            for (auto &BB : F) {
                DenseSet<Value *> Defined;
                for (auto &Inst : BB) {
                    if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
                        Value *B = BinOp->getOperand(0);
                        Value *C = BinOp->getOperand(1);
                        unsigned int opcode = BinOp->getOpcode();
                        if (BinOp->isCommutative()) {
                            tie(B, C) = minmax(B, C);
                        }
                        Expression exprKey = make_tuple(opcode, B, C);
                        unsigned int bitIndex = ExpressionToBitMap[exprKey];
    
                        // If X or Y is defined earlier in this block, it goes to Kill set
                        if (Defined.count(B) || Defined.count(C)) {
                            Kill[&BB].set(bitIndex);
                        } else {
                            Gen[&BB].set(bitIndex);
                        }
                    }
                    // Track defined values to update Kill set
                    if (!isa<BinaryOperator>(Inst)) {
                        Defined.insert(&Inst);
                    }
                }
            }

            // Initialize IN and OUT sets
            for(auto &BB : F) {
                IN[&BB] = BitVector(nextBitIndex, true);
                OUT[&BB] = BitVector(nextBitIndex, false);
            }
            
            unsigned int numExpressions = nextBitIndex;
            // Run Analsysis
            bool changed;
            do {
                changed = false;

                // First update OUT[B] for all blocks
                for (auto &BB : F) {
                    BitVector oldOUT = OUT[&BB];

                    // Compute OUT[B] = Intersection of IN[S] for all successors S
                    if (succ_begin(&BB) != succ_end(&BB)) {
                        OUT[&BB] = BitVector(numExpressions, true); // Start with universal set

                        for (auto *Succ : successors(&BB)) {
                            OUT[&BB] &= IN[Succ];  // Intersection with successor's IN set
                        }
                    }

                    // Check if OUT[B] changed
                    if (OUT[&BB] != oldOUT) {
                        changed = true;
                    }
                }

                // Then update IN[B] for all blocks
                for (auto &BB : F) {
                    // Compute IN[B] = Gen[B] ∪ (OUT[B] - Kill[B])
                    BitVector complementKill = Kill[&BB];
                    complementKill.flip();
                    BitVector temp = OUT[&BB];
                    temp &= complementKill;
                    temp |= Gen[&BB];
                    IN[&BB] = temp;
                }

            } while (changed);
            outs() << "Data Flow Analysis Completed\n";
            
            // After dataflow analysis, handle hoisting and elimination

    BasicBlock &entryBB = F.getEntryBlock();

    

    // Find the first anticipated expression (bit 0)

    Expression exprToHoist;
    bool foundExpr = false;
    for (const auto &pair : ExpressionToBitMap) {
        if (IN[&entryBB][pair.second]) {
            exprToHoist = pair.first;
            foundExpr = true;
            break;
        }
    }


    if (foundExpr) {

        // Create builder for insertion point
        IRBuilder<> Builder(&entryBB.front());
        unsigned opcode = get<0>(exprToHoist);
        Value *op1 = get<1>(exprToHoist);
        Value *op2 = get<2>(exprToHoist);


        // Create hoisted instruction
        Instruction *hoistedInst = BinaryOperator::Create(
            (Instruction::BinaryOps)opcode,
            op1,
            op2,
            "hoisted"
        );

        Builder.Insert(hoistedInst);

        // Replace uses carefully
        SmallVector<Instruction*, 8> toRemove;
        for (auto &BB : F) {
            for (auto &I : BB) {

                if (auto *binOp = dyn_cast<BinaryOperator>(&I)) {
                    if (binOp != hoistedInst && // Don't replace the hoisted instruction
                        binOp->getOpcode() == opcode) {
                        Value *B = binOp->getOperand(0);
                        Value *C = binOp->getOperand(1);

                        if (binOp->isCommutative()) {
                            tie(B, C) = minmax(B, C);
                        }

                        

                        if (B == op1 && C == op2) {
                            binOp->replaceAllUsesWith(hoistedInst);
                            toRemove.push_back(binOp);
                        }

                    }

                }

            }

        }


        // Remove redundant instructions
        for (auto *I : toRemove) {
            I->eraseFromParent();
        }


        // Write the transformed IR to a file

        std::error_code EC;
        raw_fd_ostream OS("transformed.ll", EC);
        F.getParent()->print(OS, nullptr);

    }

            // Print Gen, Kill, In, Out sets
            for (auto &BB : F) {
                errs() << "BasicBlock: " << BB.getName() << "\n";
                errs() << "  Gen:  ";
                for (unsigned i = 0; i < nextBitIndex; ++i) {
                    errs() << Gen[&BB][i];
                }
                errs() << "\n  Kill: ";
                for (unsigned i = 0; i < nextBitIndex; ++i) {
                    errs() << Kill[&BB][i];
                }
                errs() << "\n  In:   ";
                for (unsigned i = 0; i < nextBitIndex; ++i) {
                    errs() << IN[&BB][i];
                }
                errs() << "\n  Out:  ";
                for (unsigned i = 0; i < nextBitIndex; ++i) {
                    errs() << OUT[&BB][i];
                }
                errs() << "\n";
            }

            
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
