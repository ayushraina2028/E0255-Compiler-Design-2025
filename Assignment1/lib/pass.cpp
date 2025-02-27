#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"   
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/ADT/DenseMap.h"
#include <bits/stdc++.h>
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


class Expression {
public:
    enum ExprType {
        BINARY_OP,
        MEMORY_OP,
        CALL_OP,
        COMPARE_OP,
        PHI_OP,
        STORE_OP  // Added store operation type
    };

private:
    void initializeNull() {
        exprType = BINARY_OP;  // Default type
        opcode = 0;
        operand1 = nullptr;
        operand2 = nullptr;
        type = nullptr;
        callInst = nullptr;
        loadInst = nullptr;
        storeInst = nullptr;
    }

    ExprType exprType;
    unsigned opcode;
    const Value* operand1;
    const Value* operand2;
    Type* type;
    const CallInst* callInst;
    const LoadInst* loadInst;
    const StoreInst* storeInst;
    
public:
    ExprType getExprType() const { return exprType; }
    const Value* getOperand1() const { return operand1; }
    const Value* getOperand2() const { return operand2; }
    const CallInst* getCallInst() const { return callInst; }


    Expression(const Instruction* I) {
        if (!I) {
            initializeNull();
            return;
        }

        type = I->getType();
        opcode = I->getOpcode();

        if (const BinaryOperator* binOp = dyn_cast<BinaryOperator>(I)) {
            exprType = BINARY_OP;
            operand1 = binOp->getOperand(0);
            operand2 = binOp->getOperand(1);
        }
        else if (const CmpInst* cmpOp = dyn_cast<CmpInst>(I)) {
            exprType = COMPARE_OP;
            operand1 = cmpOp->getOperand(0);
            operand2 = cmpOp->getOperand(1);
        }
        else if (const LoadInst* loadOp = dyn_cast<LoadInst>(I)) {
            exprType = MEMORY_OP;
            operand1 = loadOp->getPointerOperand();
            loadInst = loadOp;
        }
        else if (const StoreInst* storeOp = dyn_cast<StoreInst>(I)) {
            exprType = STORE_OP;
            operand1 = storeOp->getValueOperand();    // Value being stored
            operand2 = storeOp->getPointerOperand();  // Location where value is stored
            storeInst = storeOp;
        }
        else if (const CallInst* callOp = dyn_cast<CallInst>(I)) {
            exprType = CALL_OP;
            callInst = callOp;
        }
        else {
            initializeNull();
        }
    }

    bool isValid() const {
        switch(exprType) {
            case BINARY_OP:
                return opcode != 0 && operand1 && operand2;
            case COMPARE_OP:
                return opcode != 0 && operand1 && operand2;
            case MEMORY_OP:
                return loadInst != nullptr;
            case STORE_OP:
                return storeInst != nullptr && operand1 && operand2;
            case CALL_OP:
                return callInst != nullptr;
            default:
                return false;
        }
    }

    void print(raw_ostream &OS) const {
        if (!isValid()) {
            OS << "Invalid Expression\n";
            return;
        }

        OS << "Expression Type: ";
        switch(exprType) {
            case BINARY_OP:
                OS << "Binary Operation\n";
                OS << "Opcode: " << opcode << "\n";
                OS << "Operand1: " << *operand1 << "\n";
                OS << "Operand2: " << *operand2 << "\n";
                break;
            case COMPARE_OP:
                OS << "Compare Operation\n";
                OS << "Opcode: " << opcode << "\n";
                OS << "Operand1: " << *operand1 << "\n";
                OS << "Operand2: " << *operand2 << "\n";
                break;
            case MEMORY_OP:
                OS << "Memory Load Operation\n";
                OS << "Load Instruction: " << *loadInst << "\n";
                break;
            case STORE_OP:
                OS << "Memory Store Operation\n";
                OS << "Value Stored: " << *operand1 << "\n";
                OS << "Store Location: " << *operand2 << "\n";
                break;
            case CALL_OP:
                OS << "Function Call\n";
                OS << "Call Instruction: " << *callInst << "\n";
                break;
            default:
                OS << "Unknown\n";
        }
        OS << "Type: " << *type << "\n";
    }

    // Update operator== to handle store operations
    bool operator==(const Expression &other) const {
        if (exprType != other.exprType) return false;
        if (opcode != other.opcode) return false;
        
        switch(exprType) {
            case BINARY_OP:
            case COMPARE_OP:
                return operand1 == other.operand1 && 
                       operand2 == other.operand2 &&
                       type == other.type;
            case MEMORY_OP:
                return loadInst == other.loadInst;
            case STORE_OP:
                return storeInst == other.storeInst &&
                       operand1 == other.operand1 &&
                       operand2 == other.operand2;
            case CALL_OP:
                return callInst == other.callInst;
            default:
                return false;
        }
    }

    // Update operator< to handle store operations
    bool operator<(const Expression &other) const {
        if (exprType != other.exprType)
            return exprType < other.exprType;
        if (opcode != other.opcode)
            return opcode < other.opcode;
            
        switch(exprType) {
            case BINARY_OP:
            case COMPARE_OP:
                if (operand1 != other.operand1)
                    return operand1 < other.operand1;
                if (operand2 != other.operand2)
                    return operand2 < other.operand2;
                return type < other.type;
            case MEMORY_OP:
                return loadInst < other.loadInst;
            case STORE_OP:
                if (operand1 != other.operand1)
                    return operand1 < other.operand1;
                if (operand2 != other.operand2)
                    return operand2 < other.operand2;
                return storeInst < other.storeInst;
            case CALL_OP:
                return callInst < other.callInst;
            default:
                return false;
        }
    }
};


struct AnticipatedExpressions : public PassInfoMixin<AnticipatedExpressions> {

    // Data structures for analysis

    using ExpressionSet = std::set<Expression>;
    using BlockExprMap = DenseMap<const BasicBlock*, ExpressionSet>;

    // Analysis information

    BlockExprMap inSets;    // IN sets for each block
    BlockExprMap outSets;   // OUT sets for each block

    // Helper methods for different patterns
    bool isLoopInvariant(const Expression& E, const Loop* L) {
        // First, let's check based on expression type
        switch(E.getExprType()) {
            case Expression::ExprType::BINARY_OP: {

            // For binary operations, check both operands
            const Value* op1 = E.getOperand1();
            const Value* op2 = E.getOperand2();

        
            // Helper lambda to check if a value is loop invariant
            auto isValueLoopInvariant = [L](const Value* V) {
                // Constants are always loop invariant
                if (isa<Constant>(V)) 
                    return true;
                
                // If it's an instruction, check if it's outside the loop
                if (const Instruction* I = dyn_cast<Instruction>(V)) {
                    // If instruction is not in the loop, it's invariant
                    return !L->contains(I->getParent());
                }

                // For other cases (arguments, global values), consider them invariant
                return true;
            };


            // Expression is loop invariant if both operands are
            return isValueLoopInvariant(op1) && isValueLoopInvariant(op2);
        }

            case Expression::MEMORY_OP: {
                // For load instructions, we need to check:
                // 1. If the pointer operand is loop invariant
                // 2. If there are no stores to the same location in the loop
                const Value* ptr = E.getOperand1();
                
                // First check if pointer is loop invariant
                if (const Instruction* I = dyn_cast<Instruction>(ptr)) {
                    if (L->contains(I->getParent()))
                        return false;
                }

                // Conservative approach: if there's any store in the loop,
                // consider the load as not invariant
                for (const BasicBlock* BB : L->blocks()) {
                    for (const Instruction& I : *BB) {
                        if (isa<StoreInst>(I))
                            return false;
                    }
                }
                return true;
            }

            case Expression::STORE_OP: {
                // Stores are generally not considered loop invariant
                // as they represent side effects
                return false;
            }

            case Expression::CALL_OP: {
                // For calls, check if it's a pure function
                // (this is a conservative approach)
                if (const CallInst* CI = E.getCallInst()) {
                    // If the function only reads memory and has no other side effects
                    return CI->onlyReadsMemory();
                }
                return false;
            }

            default:
                // For any other type of expression, take conservative approach
                return false;
        }
    }

    bool mayKillExpression(const Instruction& I, const Expression& E) {

        // For store instructions
        if (const StoreInst* SI = dyn_cast<StoreInst>(&I)) {
            const Value* storedTo = SI->getPointerOperand();

            switch(E.getExprType()) {
                case Expression::MEMORY_OP: {
                    // If we're storing to the same location that's being loaded
                    if (E.getOperand1() == storedTo) return true;
                    break;
                }

                case Expression::BINARY_OP: {
                    // If we're storing to a location that's used in the binary operation
                    if (E.getOperand1() == storedTo || E.getOperand2() == storedTo)
                        return true;
                    break;
                }

                case Expression::STORE_OP: {
                    // If we're storing to the same location
                    if (E.getOperand2() == storedTo) return true;
                    break;

                }
            }
        }

        

        // For call instructions
        if (const CallInst* CI = dyn_cast<CallInst>(&I)) {

            // Conservative approach: if the call can write to memory,
            // assume it might modify any memory location
            if (!CI->onlyReadsMemory()) {
                if (E.getExprType() == Expression::MEMORY_OP)
                    return true;
            }
        }
        
        return false;
    }
    bool isRedundantExpression(const Expression& E, const BasicBlock* BB) {

        // Count occurrences of the same expression in the basic block
        int count = 0;

        // First pass: Check current basic block
        for (const Instruction& I : *BB) {
            Expression currExpr(&I);
            if (currExpr == E) {
                count++;
                if (count > 1) {
                    // Found redundant computation in the same block
                    return true;
                }
            }
        }

        // Second pass: Check predecessor blocks
        for (const BasicBlock* Pred : predecessors(BB)) {
            for (const Instruction& I : *Pred) {
                Expression currExpr(&I);
                if (currExpr == E) {
                    // Found same expression in predecessor
                    // Now check if the expression's value is still valid
                    bool isKilled = false;
                    // Check if any instruction between this expression and the end 
                    // of predecessor block modifies the operands
                    for (const Instruction& PredI : make_range(++I.getIterator(), Pred->end())) {
                        if (mayKillExpression(PredI, E)) {
                            isKilled = true;
                            break;
                        }
                    }


                    // If not killed in predecessor, check current block up to the expression
                    if (!isKilled) {
                        for (const Instruction& CurrI : *BB) {
                            if (mayKillExpression(CurrI, E)) {
                                isKilled = true;
                                break;
                            }
                        }
                    }

                    if (!isKilled) {
                        return true;  // Found redundant computation
                    }
                }
            }
        }
        return false;
    }

    bool isAnticipatedInBranch(const Expression& E, const BasicBlock* BB) {
        // Get the terminator instruction of the block
        const Instruction* terminator = BB->getTerminator();
        const BranchInst* brInst = dyn_cast<BranchInst>(terminator);

        // If not a conditional branch, return false
        if (!brInst || !brInst->isConditional()) {
            return false;
        }


        // Get the true and false branches
        const BasicBlock* trueBB = brInst->getSuccessor(0);
        const BasicBlock* falseBB = brInst->getSuccessor(1);

        // Helper function to check if expression appears before any modification

        auto isExpressionAnticipatedInBlock = [this](const Expression& expr, const BasicBlock* block) {

            for (const Instruction& I : *block) {

                // If we find the expression first, it's anticipated
                Expression currExpr(&I);
                if (currExpr == expr) {
                    return true;
                }

                // If we find an instruction that kills the expression, it's not anticipated
                if (mayKillExpression(I, expr)) {
                    return false;
                }
            }

            return false;
        };


        // Check if the expression is anticipated in both paths

        bool anticipatedInTrue = isExpressionAnticipatedInBlock(E, trueBB);
        bool anticipatedInFalse = isExpressionAnticipatedInBlock(E, falseBB);

        // Expression is anticipated if:
        // 1. It appears in both branches before any modification, or
        // 2. It appears in one branch and the other branch is empty/returns
        if (anticipatedInTrue && anticipatedInFalse) {
            return true;
        }


        // Check for special cases where one branch leads to immediate return/exit
        auto isImmediateExit = [](const BasicBlock* block) {
            return block->size() == 1 && (isa<ReturnInst>(block->getTerminator()) ||
                                        isa<UnreachableInst>(block->getTerminator()));
        };


        if ((anticipatedInTrue && isImmediateExit(falseBB)) ||
            (anticipatedInFalse && isImmediateExit(trueBB))) {
            return true;
        }

        // Additional check for expressions used in the branch condition itself
        const Value* condition = brInst->getCondition();
        if (const Instruction* condInst = dyn_cast<Instruction>(condition)) {
            Expression condExpr(condInst);
            if (condExpr == E) {
                return true;
            }
        }

        return false;

    }
    bool isExpressionInMultipleCases(const Expression& E, const SwitchInst* SI) {
        int caseCount = 0;
        for (auto& caseHandle : SI->cases()) {
            const BasicBlock* caseBB = caseHandle.getCaseSuccessor();
            // Check each instruction in the case block
            for (const Instruction& I : *caseBB) {
                if (I.isTerminator()) break;
                
                Expression caseExpr(&I);

                if (caseExpr == E) {
                    caseCount++;
                    if (caseCount > 1) {
                        return true;  // Found in multiple cases
                    }
                    break;  // Move to next case
                }

                // Stop if we hit an instruction that would kill the expression
                if (mayKillExpression(I, E)) {
                    break;
                }
            }
        }
        
        return false;
    }

    ExpressionSet getExpressionsInSwitch(const SwitchInst* SI) {
        ExpressionSet expressions;

        // First, add the switch condition expression
        if (const Instruction* condInst = dyn_cast<Instruction>(SI->getCondition())) {
            Expression condExpr(condInst);
            if (condExpr.isValid()) {
                expressions.insert(condExpr);
            }
        }

        // Get the default case
        const BasicBlock* defaultBB = SI->getDefaultDest();
        
        // Analyze each case
        for (auto& caseHandle : SI->cases()) {
            // Get the case value and destination block
            const BasicBlock* caseBB = caseHandle.getCaseSuccessor();

            // Skip if this is the default case
            if (caseBB == defaultBB) continue;

            // Analyze expressions in the case block
            for (const Instruction& I : *caseBB) {
                // Stop if we hit a terminator or branch
                if (I.isTerminator()) break;
                Expression expr(&I);
                if (expr.isValid()) {
                    // Check if this expression uses the switch condition
                    bool usesCondition = false;
                    if (const Instruction* condInst = dyn_cast<Instruction>(SI->getCondition())) {
                        if (expr.getOperand1() == condInst || expr.getOperand2() == condInst) {
                            usesCondition = true;
                        }
                    }
                
                    // Add expressions that:

                    // 1. Use the switch condition

                    // 2. Appear in multiple cases

                    // 3. Are not modified before use

                    if (usesCondition || isExpressionInMultipleCases(expr, SI)) {
                        expressions.insert(expr);
                    }
                }
            }
        }

        return expressions;
    }

    bool isMathFunctionCall(const CallInst* CI) {
        if (!CI) return false;

        // Get the called function
        const Function* calledFunc = CI->getCalledFunction();
        if (!calledFunc) return false;

        // Get the function name
        StringRef funcName = calledFunc->getName();

        // Common math library functions
        static const std::set<StringRef> mathFuncs = {
            // Basic math functions
            "sqrt", "pow", "exp", "log", "log10", "log2",
            // Trigonometric functions
            "sin", "cos", "tan",
            "asin", "acos", "atan", "atan2",
            // Hyperbolic functions
            "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh",
            // Rounding and absolute value
            "ceil", "floor", "round", "trunc",
            "abs", "fabs",
            // Float manipulation
            "fmod", "remainder",
            // Special functions
            "erf", "erfc", "gamma", "lgamma",
            // LLVM intrinsic math functions
            "llvm.sqrt", "llvm.pow", "llvm.exp", "llvm.log",
            "llvm.sin", "llvm.cos", "llvm.floor", "llvm.ceil"
        };

        // Check standard math functions
        if (mathFuncs.count(funcName)) {
            return true;
        }

        // Check if it's an LLVM math intrinsic
        if (funcName.contains("llvm.math")) {
            return true;
        }

        // Additional checks for specific properties
        if (calledFunc->onlyReadsMemory() && // Pure function
            calledFunc->getReturnType()->isFloatingPointTy() && // Returns floating point
            calledFunc->arg_size() <= 4) { // Takes reasonable number of arguments
            // Likely a math function
            return true;
        }

        return false;
    }
    // Analysis methods

    void analyzeIfElsePattern(const BranchInst* BI) {
        if (!BI || !BI->isConditional()) return;

        // Get the condition and branches
        const Value* condition = BI->getCondition();
        const BasicBlock* trueBB = BI->getSuccessor(0);
        const BasicBlock* falseBB = BI->getSuccessor(1);

        ExpressionSet anticipatedExprs;

        // Analyze the condition itself
        if (const CmpInst* cmp = dyn_cast<CmpInst>(condition)) {
            Expression cmpExpr(cmp);
            if (cmpExpr.isValid()) {
                anticipatedExprs.insert(cmpExpr);
            }
        }

        // Collect expressions from true block
        for (const Instruction& I : *trueBB) {
            if (isa<PHINode>(I)) continue;
            Expression expr(&I);
            if (expr.isValid()) {
                anticipatedExprs.insert(expr);
            }
        }

        // Collect expressions from false block
        for (const Instruction& I : *falseBB) {
            if (isa<PHINode>(I)) continue;
            Expression expr(&I);
            if (expr.isValid()) {
                anticipatedExprs.insert(expr);
            }
        }

        // Print analysis results
        errs() << "If/Else Pattern Analysis Results:\n";
        for (const Expression& E : anticipatedExprs) {
            errs() << "  Found expression:\n    ";
            E.print(errs());
            errs() << "\n";
        }
    }

    void analyzeLoop(const Loop* L) {
        if (!L) return;

        ExpressionSet loopExpressions;
        ExpressionSet invariantExpressions;

        // Get loop header
        BasicBlock* header = L->getHeader();
        
        // Analyze loop header
        for (const Instruction& I : *header) {
            Expression expr(&I);
            if (expr.isValid()) {
                loopExpressions.insert(expr);
                
                // Check if expression is loop invariant
                if (isLoopInvariant(expr, L)) {
                    invariantExpressions.insert(expr);
                }
            }
        }

        // Analyze loop body blocks
        for (const BasicBlock* BB : L->blocks()) {
            if (BB == header) continue; // Skip header as we've already analyzed it

            for (const Instruction& I : *BB) {
                if (isa<PHINode>(I)) continue; // Skip PHI nodes
                
                Expression expr(&I);
                if (!expr.isValid()) continue;

                loopExpressions.insert(expr);

                // Check for loop invariant expressions
                if (isLoopInvariant(expr, L)) {
                    invariantExpressions.insert(expr);
                }
            }
        }

        // Print analysis results
        errs() << "Loop Analysis Results:\n";
        errs() << "Loop Header: " << header->getName() << "\n";
        
        errs() << "Loop Expressions:\n";
        for (const Expression& E : loopExpressions) {
            errs() << "  Regular Expression:\n    ";
            E.print(errs());
            errs() << "\n";
        }

        errs() << "Loop Invariant Expressions:\n";
        for (const Expression& E : invariantExpressions) {
            errs() << "  Invariant Expression:\n    ";
            E.print(errs());
            errs() << "\n";
        }
    }

    void analyzeMemoryOperations(const BasicBlock* BB) {
        if (!BB) return;

        ExpressionSet loadExprs;
        ExpressionSet storeExprs;
        std::map<const Value*, const StoreInst*> lastStore;

        // First pass: collect all memory operations
        for (const Instruction& I : *BB) {
            Expression expr(&I);
            if (!expr.isValid()) continue;

            if (isa<LoadInst>(I)) {
                loadExprs.insert(expr);
            }
            else if (const StoreInst* SI = dyn_cast<StoreInst>(&I)) {
                storeExprs.insert(expr);
                lastStore[SI->getPointerOperand()] = SI;
            }
        }

        // Print analysis results
        errs() << "Memory Operations Analysis for Basic Block: " 
            << BB->getName() << "\n";

        // Analyze and print load operations
        errs() << "Load Operations:\n";
        for (const Expression& E : loadExprs) {
            errs() << "  Found Load:\n    ";
            E.print(errs());
            
            // Check if there's a previous store to the same location
            const Value* loadPtr = E.getOperand1(); // For loads, operand1 is the pointer
            if (lastStore.count(loadPtr)) {
                errs() << "    Has previous store in this block\n";
            } else {
                errs() << "    No previous store in this block\n";
            }
            errs() << "\n";
        }

        // Analyze and print store operations
        errs() << "Store Operations:\n";
        for (const Expression& E : storeExprs) {
            errs() << "  Found Store:\n    ";
            E.print(errs());
            errs() << "\n";
        }

        // Analyze store-load patterns
        errs() << "Store-Load Patterns:\n";
        for (const auto& pair : lastStore) {
            const Value* ptr = pair.first;
            bool hasSubsequentLoad = false;

            // Check for loads after this store
            for (const Instruction& I : *BB) {
                if (const LoadInst* LI = dyn_cast<LoadInst>(&I)) {
                    if (LI->getPointerOperand() == ptr) {
                        hasSubsequentLoad = true;
                        errs() << "  Found store-load pattern for location: "
                            << *ptr << "\n";
                        break;
                    }
                }
            }

            if (!hasSubsequentLoad) {
                errs() << "  Store without subsequent load for location: "
                    << *ptr << "\n";
            }
        }
    }

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
        errs() << "\n=== Testing Expression Creation and Printing ===\n";

        for (auto &BB : F) {
            errs() << "Basic Block: " << BB.getName() << "\n";
            for (auto &I : BB) {
                Expression expr(&I);
                if (expr.isValid()) {
                    errs() << "Found valid expression:\n";
                    expr.print(errs());
                    errs() << "\n";
                }
            }
        }
        
         errs() << "\n=== Testing Memory Operations Analysis ===\n";

        for (auto &BB : F) {
            errs() << "Analyzing Memory Operations in Block: " << BB.getName() << "\n";
            analyzeMemoryOperations(&BB);
        }

        errs() << "\n=== Testing If/Else Pattern Analysis ===\n";

        for (auto &BB : F) {
            if (auto* BI = dyn_cast<BranchInst>(BB.getTerminator())) {
                if (BI->isConditional()) {
                    errs() << "Analyzing If/Else in Block: " << BB.getName() << "\n";
                    analyzeIfElsePattern(BI);
                }
            }
        }

        // Test redundant expressions

        errs() << "\n=== Testing Redundant Expressions ===\n";
        for (auto &BB : F) {
            errs() << "Checking Redundant Expressions in Block: " << BB.getName() << "\n";
            for (auto &I : BB) {
                Expression expr(&I);
                if (expr.isValid()) {
                    if (isRedundantExpression(expr, &BB)) {
                        errs() << "Found Redundant Expression:\n";
                        expr.print(errs());
                        errs() << "\n";
                    }
                }
            }
        }
    
            // Test math function calls

        errs() << "\n=== Testing Math Function Calls ===\n";

        for (auto &BB : F) {
            errs() << "Checking Math Functions in Block: " << BB.getName() << "\n";
            for (auto &I : BB) {
                if (const CallInst* CI = dyn_cast<CallInst>(&I)) {
                    if (isMathFunctionCall(CI)) {
                        errs() << "Found Math Function Call: " 
                            << CI->getCalledFunction()->getName() << "\n";
                    }
                }
            }
        }

        // Test switch patterns
        errs() << "\n=== Testing Switch Pattern Analysis ===\n";
        for (auto &BB : F) {
            if (auto* SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
                errs() << "Analyzing Switch in Block: " << BB.getName() << "\n";
                auto switchExprs = getExpressionsInSwitch(SI);
                errs() << "Expressions in Switch:\n";
                for (const auto& expr : switchExprs) {
                    expr.print(errs());
                    errs() << "\n";
                }
            }
        }

        // Test loop analysis


        errs() << "\n=== Testing Loop Analysis ===\n";
        for (Loop* L : LI.getLoopsInPreorder()) {
            errs() << "Analyzing Loop\n";
            analyzeLoop(L);


            // Test loop invariant expressions
            errs() << "Checking Loop Invariant Expressions:\n";
            for (auto &BB : L->blocks()) {
                for (auto &I : *BB) {
                    Expression expr(&I);
                    if (expr.isValid() && isLoopInvariant(expr, L)) {
                        errs() << "Found Loop Invariant Expression:\n";
                        expr.print(errs());
                        errs() << "\n";
                    }
                }
            }
        }

        // Test anticipated expressions in branches

        errs() << "\n=== Testing Anticipated Expressions in Branches ===\n";
        for (auto &BB : F) {
            for (auto &I : BB) {
                Expression expr(&I);
                if (expr.isValid() && isAnticipatedInBranch(expr, &BB)) {
                    errs() << "Found Anticipated Expression in Branch:\n";
                    expr.print(errs());
                    errs() << "\n";
                }
            }
        }

        return PreservedAnalyses::all();
    }
    

    static bool isRequired() { return true; }

};


} // end anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "AnticipatedExpressions", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "anticipated-expressions") {
                        FPM.addPass(AnticipatedExpressions());
                        return true;
                    }
                    return false;
                });
        }
    };
}