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
        unsigned int HoistCounter = 0;
        DenseMap<Expression, Value*> HoistedExpressions;

        void printSets(Function &F, DenseMap<BasicBlock*, BitVector> &Gen, DenseMap<BasicBlock*, BitVector> &Kill, DenseMap<BasicBlock*, BitVector> &IN, DenseMap<BasicBlock*, BitVector> &OUT, unsigned int nextBitIndex) {
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
        }
    
        PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
            bool OuterChanged;
            int iteration = 0;
            HoistCounter = 0;
            
            do {
                OuterChanged = false;
                outs() << "\n=== Starting Iteration " << ++iteration << " ===\n";
                
                // Clear previous analysis results
                DenseMap<Expression, unsigned int> ExpressionToBitMap;
                unsigned int nextBitIndex = 0;
                
                DenseMap<BasicBlock*, BitVector> IN;
                DenseMap<BasicBlock*, BitVector> OUT;
                DenseMap<BasicBlock*, BitVector> Gen;
                DenseMap<BasicBlock*, BitVector> Kill;

                // Your existing analysis code here
                // Step 1: Collect unique expressions
                // In the first pass where we collect expressions
                // In the first pass
                for (auto &BB : F) {
                    for (auto &Inst : BB) {
                        if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
                            unsigned int opcode = BinOp->getOpcode();
                            Value *op1 = BinOp->getOperand(0);    
                            Value *op2 = BinOp->getOperand(1);

                            // Debug print
                            outs() << "Analyzing instruction: " << *BinOp << "\n";
                            outs() << "  op1: " << *op1 << "\n";
                            outs() << "  op2: " << *op2 << "\n";

                            // Create expression key
                            Expression exprKey = make_tuple(opcode, op1, op2);
                            
                            if (ExpressionToBitMap.find(exprKey) == ExpressionToBitMap.end()) {
                                outs() << "  Adding new expression at index " << nextBitIndex << "\n";
                                ExpressionToBitMap[exprKey] = nextBitIndex++;
                            }
                        }
                        else if(auto* Cast = dyn_cast<CastInst>(&Inst)) {
                            Value* op = Cast->getOperand(0);
                            Expression exprKey = make_tuple(Instruction::CastOps(Cast->getOpcode()), op, nullptr);

                            if (ExpressionToBitMap.find(exprKey) == ExpressionToBitMap.end()) {
                                outs() << "  Adding new expression at index " << nextBitIndex << "\n";
                                ExpressionToBitMap[exprKey] = nextBitIndex++;
                            }
                        }
                        else if(auto* Call = dyn_cast<CallInst>(&Inst)) {
                            if(Call->getCalledFunction() && Call->getCalledFunction()->getName() == "exp") {
                                Value* op = Call->getOperand(0);
                                Expression exprKey = make_tuple(Instruction::Call, op, nullptr);

                                if (ExpressionToBitMap.find(exprKey) == ExpressionToBitMap.end()) {
                                    outs() << "  Adding new expression at index " << nextBitIndex << "\n";
                                    ExpressionToBitMap[exprKey] = nextBitIndex++;
                                }
                            }
                        }
                    }
                }
                
                // Step 2 & 3: Your existing Gen/Kill computation
                // ... (keep your existing code)
                // Step 2: Initialize Gen and Kill sets for each basic block
                for (auto &BB : F) {
                    Gen[&BB] = BitVector(nextBitIndex, false);
                    Kill[&BB] = BitVector(nextBitIndex, false);
                }

                 // Step 3: Compute Gen and Kill sets
                for (auto &BB : F) {
                    DenseSet<Value *> Defined;
                    for (auto &Inst : BB) {
                        // In Gen/Kill computation
                        if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
                            Value *op1 = BinOp->getOperand(0);
                            Value *op2 = BinOp->getOperand(1);
                            unsigned int opcode = BinOp->getOpcode();
                            
                            Expression exprKey = make_tuple(opcode, op1, op2);
                            auto it = ExpressionToBitMap.find(exprKey);
                            if (it != ExpressionToBitMap.end()) {
                                unsigned int bitIndex = it->second;
                                if (Defined.count(op1) || Defined.count(op2)) {
                                    Kill[&BB].set(bitIndex);
                                } else {
                                    Gen[&BB].set(bitIndex);
                                }
                            }
                        }
                        else if (auto *Cast = dyn_cast<CastInst>(&Inst)) {

                        Value *op = Cast->getOperand(0);

                        Expression exprKey = make_tuple(Cast->getOpcode(), op, nullptr);

                        auto it = ExpressionToBitMap.find(exprKey);

                        if (it != ExpressionToBitMap.end()) {

                            unsigned int bitIndex = it->second;

                            if (Defined.count(op)) {

                                Kill[&BB].set(bitIndex);

                            } else {

                                Gen[&BB].set(bitIndex);

                            }

                        }

                    } else if (auto *Call = dyn_cast<CallInst>(&Inst)) {

                        if (Call->getCalledFunction() && 

                            Call->getCalledFunction()->getName() == "exp") {

                            Value *op = Call->getArgOperand(0);

                            Expression exprKey = make_tuple(Instruction::Call, op, nullptr);

                            auto it = ExpressionToBitMap.find(exprKey);

                            if (it != ExpressionToBitMap.end()) {

                                unsigned int bitIndex = it->second;

                                if (Defined.count(op)) {

                                    Kill[&BB].set(bitIndex);

                                } else {

                                    Gen[&BB].set(bitIndex);

                                }

                            }

                        }

                    }
                        // Track defined values to update Kill set
                        Defined.insert(&Inst);
                    }
                }

                // Step 4: Your existing dataflow analysis
                // ... (keep your existing code)
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
                
                

                outs() << "\nData Flow Analysis Completed for Iteration " << iteration << "\n";
                
                // Print Gen, Kill, In, Out sets
                printSets(F, Gen, Kill, IN, OUT, nextBitIndex);

                outs() << "\nIR before hoisting (Iteration " << iteration << "):\n";
                F.print(outs());

                // Get DominatorTree and perform hoisting
                DominatorTree DT(F);
                
                // Store number of anticipated expressions before hoisting
                // Store IR state before hoisting

                std::string beforeIR;

                raw_string_ostream rso(beforeIR);

                F.print(rso);

                int anticipatedBefore = 0;
                for (auto &BB : F) {
                    BitVector &anticipatedExprs = OUT[&BB];
                    anticipatedBefore += anticipatedExprs.count();
                }

                outs() << "Anticipated expressions before hoisting: " << anticipatedBefore << "\n";


                int replacementsMade = 0;

                if(anticipatedBefore > 0) {

                    replacementsMade = hoistAnticipatedExpressions(F, OUT, ExpressionToBitMap, DT);

                    OuterChanged = (replacementsMade > 0);  // Only continue if we actually made replacements

                }

                outs() << "\nIR after hoisting (Iteration " << iteration << "):\n";
                F.print(outs());

                outs() << "Replacements made: " << replacementsMade << "\n";
                outs() << "=== Iteration " << iteration << " completed ===\n";

            } while (OuterChanged);

            outs() << "\n=== Optimization completed after " << iteration << " iterations ===\n";
            
            std::error_code EC;
            raw_fd_ostream file("modified_output.ll", EC);
            F.print(file, nullptr);

            return PreservedAnalyses::none();
        }

        int hoistAnticipatedExpressions(Function &F, 
                               DenseMap<BasicBlock*, BitVector> &OUT,
                               DenseMap<Expression, unsigned int> &ExpressionToBitMap,
                               DominatorTree &DT) {
    
    int totalReplacements = 0;
    outs() << "\n=== Starting Hoisting Process ===\n";
    
    DenseMap<unsigned int, Expression> BitToExpression;
    for (const auto &Entry : ExpressionToBitMap) {
        BitToExpression[Entry.second] = Entry.first;
    }

    DenseMap<Expression, Value*> HoistedExpressions;
    DenseMap<Value*, Value*> ReplacementMap;

    for (auto &BB : F) {
        outs() << "\nProcessing Block: " << BB.getName() << "\n";
        BitVector &anticipatedExprs = OUT[&BB];
        
        outs() << "OUT set for this block: ";
        for (unsigned i = 0; i < anticipatedExprs.size(); ++i) {
            outs() << anticipatedExprs[i];
        }
        outs() << "\n";

        SmallVector<std::pair<unsigned, Expression>, 8> ExpressionsToHoist;
        for (unsigned i = 0; i < anticipatedExprs.size(); ++i) {
            if (anticipatedExprs[i]) {
                Expression expr = BitToExpression[i];
                ExpressionsToHoist.push_back({i, expr});
            }
        }

        std::sort(ExpressionsToHoist.begin(), ExpressionsToHoist.end(),
                 [](const auto &a, const auto &b) {
                     return a.first < b.first;
                 });

        for (const auto &ExprPair : ExpressionsToHoist) {
            unsigned i = ExprPair.first;
            Expression expr = ExprPair.second;
            
            if (HoistedExpressions.count(expr) > 0) {
                outs() << "Expression already hoisted, skipping...\n";
                continue;
            }

            unsigned opcode = get<0>(expr);
            Value *op1 = get<1>(expr);
            Value *op2 = get<2>(expr);

            if (ReplacementMap.count(op1)) {
                op1 = ReplacementMap[op1];
            }
            if (ReplacementMap.count(op2)) {
                op2 = ReplacementMap[op2];
            }

            outs() << "\nProcessing expression at bit " << i << ":\n";
            outs() << "  Original instruction string: ";
            if (auto *I = dyn_cast<Instruction>(op1)) {
                I->print(outs());
            }
            outs() << "\n";

            outs() << "Found anticipated expression at bit " << i << "\n";
            outs() << "Opcode: " << Instruction::getOpcodeName(opcode) << "\n";
            outs() << "Operand 1: " << *op1 << "\n";
            
            if(opcode != Instruction::Call && opcode != Instruction::SIToFP) {
                outs() << "Operand 2: " << *op2 << "\n";
            }

            outs() << "Hoisting to block: " << BB.getName() << "\n";

            std::string hoistedName = "hoisted" + std::to_string(++HoistCounter);
            IRBuilder<> Builder(&BB, BB.getTerminator()->getIterator());
            
            Value *newInst;
            if (opcode == Instruction::SRem) {
                newInst = Builder.CreateSRem(op1, op2, hoistedName);
            } 
            else if(opcode == Instruction::SIToFP) {
                newInst = Builder.CreateSIToFP(op1, Type::getDoubleTy(F.getContext()), hoistedName);
            }
            else if(opcode == Instruction::Call) {
                Function *ExpFunc = F.getParent()->getFunction("exp");
                newInst = Builder.CreateCall(ExpFunc, op1, hoistedName);
            }
            else {
                newInst = Builder.CreateBinOp(
                    static_cast<Instruction::BinaryOps>(opcode),
                    op1, op2, hoistedName);
            }

            if (BinaryOperator* binOp = dyn_cast<BinaryOperator>(newInst)) {
                binOp->setHasNoSignedWrap(true);
            }
            
            HoistedExpressions[expr] = newInst;
            outs() << "Created new instruction: " << *newInst << "\n";

            // Replace this part in your code
            SmallVector<Instruction*, 8> InstToReplace;
            for (auto &Block : F) {
                for (auto &I : Block) {
                    if (auto *binOp = dyn_cast<BinaryOperator>(&I)) {
                        if (binOp == newInst) continue;
                        
                        Value *curOp1 = binOp->getOperand(0);
                        Value *curOp2 = binOp->getOperand(1);
                        
                        if (ReplacementMap.count(curOp1)) {
                            curOp1 = ReplacementMap[curOp1];
                        }
                        if (ReplacementMap.count(curOp2)) {
                            curOp2 = ReplacementMap[curOp2];
                        }

                        if (binOp->getOpcode() == opcode &&
                            ((curOp1 == op1 && curOp2 == op2) ||
                                (binOp->isCommutative() && 
                                curOp1 == op2 && curOp2 == op1))) {
                            InstToReplace.push_back(binOp);
                        }
                    }
                    // Add handling for CastInst
                    else if (auto *castInst = dyn_cast<CastInst>(&I)) {
                        if (castInst == newInst) continue;
                        
                        Value *curOp = castInst->getOperand(0);
                        if (ReplacementMap.count(curOp)) {
                            curOp = ReplacementMap[curOp];
                        }

                        if (castInst->getOpcode() == opcode && curOp == op1) {
                            InstToReplace.push_back(castInst);
                        }
                    }
                    // Add handling for CallInst
                    else if (auto *callInst = dyn_cast<CallInst>(&I)) {
                        if (callInst == newInst) continue;
                        
                        if (callInst->getCalledFunction() && 
                            callInst->getCalledFunction()->getName() == "exp") {
                            Value *curOp = callInst->getArgOperand(0);
                            if (ReplacementMap.count(curOp)) {
                                curOp = ReplacementMap[curOp];
                            }

                            if (opcode == Instruction::Call && curOp == op1) {
                                InstToReplace.push_back(callInst);
                            }
                        }
                    }
                }
            }

            outs() << "\nReplacing instructions:\n";
            for (auto *I : InstToReplace) {
                outs() << "  Replacing: " << *I << "\n";
                ReplacementMap[I] = newInst;
                I->replaceAllUsesWith(newInst);
                I->eraseFromParent();
                outs() << "  Replacement complete\n";
                totalReplacements++;
            }
            outs() << "Total replacements: " << InstToReplace.size() << "\n";
        }
    }
    
    outs() << "\n=== Hoisting Process Complete ===\n";
    return totalReplacements;
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
