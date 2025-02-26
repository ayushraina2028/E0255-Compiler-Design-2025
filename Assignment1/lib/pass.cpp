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

    struct Expression {
        Value* LHS;
        Value* RHS;
        unsigned Opcode;

        Expression(Value* LHS, Value* RHS, unsigned Opcode) 
            : LHS(LHS), RHS(RHS), Opcode(Opcode) {}

        bool operator==(const Expression& other) const {
            return LHS == other.LHS && RHS == other.RHS && Opcode == other.Opcode;
        }

        bool operator<(const Expression& other) const {
            if (LHS != other.LHS) return LHS < other.LHS;
            if (RHS != other.RHS) return RHS < other.RHS;
            return Opcode < other.Opcode;
        }

        void dump() const {
            errs() << "Expression: " << *LHS << " " 
                   << Instruction::getOpcodeName(Opcode) << " " 
                   << *RHS << "\n";
        }
    };


    struct MemoryState {

        Value* LastStoredValue;    // Last value stored to this location
        Instruction* LastStoreInst; // Last store instruction
        bool isInitialized;        // Whether the location has been initialized
        set<LoadInst*> LoadUsers;  // All loads reading from this location

        MemoryState() : LastStoredValue(nullptr), LastStoreInst(nullptr), 
                       isInitialized(false) {}

    };

    using ExpressionSet = set<Expression>;
    DenseMap<BasicBlock *, ExpressionSet> IN;
    DenseMap<BasicBlock *, ExpressionSet> OUT;
    DenseMap<BasicBlock *, ExpressionSet> V_USE;
    DenseMap<BasicBlock *, ExpressionSet> V_DEF;
    ExpressionSet UniversalSet;


    // Memory tracking structures
    DenseMap<Value*, MemoryState> MemoryStates;  // Track complete memory state
    DenseMap<Value*, Value*> LoadedValues;       // Maps loaded values to their memory locations
    DenseMap<Value*, Value*> ValueOrigins;       // Track where values originally came from


    // Helper function to get the actual memory location
    Value* getUnderlyingObject(Value* V) {
        if (GetElementPtrInst* GEP = dyn_cast<GetElementPtrInst>(V)) {
            return getUnderlyingObject(GEP->getPointerOperand());
        }

        if (BitCastInst* BC = dyn_cast<BitCastInst>(V)) {
            return getUnderlyingObject(BC->getOperand(0));
        }

        if (LoadInst* LI = dyn_cast<LoadInst>(V)) {
            return getUnderlyingObject(LI->getPointerOperand());
        }

        return V;
    }


    // Helper function to check program order
    bool isProgramOrderBefore(Instruction* I1, Instruction* I2) {
        if (I1->getParent() != I2->getParent()) 
            return false;

        for (Instruction& I : *I1->getParent()) {
            if (&I == I1) return true;
            if (&I == I2) return false;
        }

        return false;
    }


    void handleStore(StoreInst* SI) {
        Value* ValueStored = SI->getValueOperand();
        Value* Location = SI->getPointerOperand();

        // Get the actual location
        Location = getUnderlyingObject(Location);

        // Update memory state
        MemoryState& State = MemoryStates[Location];
        State.LastStoredValue = ValueStored;
        State.LastStoreInst = SI;
        State.isInitialized = true;

        // Track the origin of the stored value
        if (Instruction* I = dyn_cast<Instruction>(ValueStored)) {
            ValueOrigins[Location] = I;
        }


        // Update all loads that might be affected by this store
        for (LoadInst* LI : State.LoadUsers) {
            if (LI->getParent() == SI->getParent() && 
                !isProgramOrderBefore(LI, SI)) {
                LoadedValues[LI] = ValueStored;
            }
        }


        errs() << "Store handled: " << *SI << "\n";
        errs() << "  Location: " << *Location << "\n";
        errs() << "  Value: " << *ValueStored << "\n";
    }


    void handleLoad(LoadInst* LI) {
    Value* Location = LI->getPointerOperand();
    
    // Get the actual location
    Location = getUnderlyingObject(Location);

    // If loading from an uninitialized alloca, mark as undefined
    if (auto* AI = dyn_cast<AllocaInst>(Location)) {
        bool hasStore = false;
        for (Instruction &PrevInst : *LI->getParent()) {
            if (&PrevInst == LI) 
                break;
            if (auto* SI = dyn_cast<StoreInst>(&PrevInst)) {
                if (SI->getPointerOperand() == AI) {
                    hasStore = true;
                    break;
                }
            }
        }
        if (!hasStore) {
            LoadedValues[LI] = nullptr;
            return;
        }
    }

    // Update memory state
    MemoryState& State = MemoryStates[Location];
    State.LoadUsers.insert(LI);

    // If we have a known value stored here, track it
    if (State.isInitialized && State.LastStoredValue) {
        LoadedValues[LI] = State.LastStoredValue;
        
        // Track the origin
        if (Value* Origin = ValueOrigins[Location]) {
            ValueOrigins[LI] = Origin;
        }
    }

    errs() << "Load handled: " << *LI << "\n";
    errs() << "  Location: " << *Location << "\n";
    if (State.LastStoredValue) {
        errs() << "  Last stored value: " << *State.LastStoredValue << "\n";
    }
}


    Value* getActualValue(Value* V) {
        if (auto* LI = dyn_cast<LoadInst>(V)) {
            Value* Location = getUnderlyingObject(LI->getPointerOperand());
            auto& State = MemoryStates[Location];

            if (State.isInitialized && State.LastStoredValue) {
                if (State.LastStoreInst && 
                    isProgramOrderBefore(State.LastStoreInst, LI)) {
                    return State.LastStoredValue;
                }
            }

            if (Value* Origin = ValueOrigins[V]) {
                return Origin;
            }

        }
        return V;
    }


    bool isDefinedInInstruction(Value* V, Instruction& I) {
    // Check if this instruction directly defines V
    if (dyn_cast<Value>(&I) == V)
        return true;

    // Check store instructions
    if (auto* SI = dyn_cast<StoreInst>(&I)) {
        Value* Location = SI->getPointerOperand();
        // Check if V is loaded from this location
        if (auto* LI = dyn_cast<LoadInst>(V)) {
            return LoadedValues[LI] == Location;
        }
        // Check if V is the value being stored
        if (SI->getValueOperand() == V) {
            return true;
        }
    }

    return false;
}


    bool isDefinedBefore(Value* V, Instruction &I, BasicBlock &BB) {

    errs() << "Checking if " << *V << " is defined before " << I << "\n";
    // If V is an argument or constant, it's not defined in this block
    if (isa<Argument>(V) || isa<Constant>(V))
        return false;

    // If V is an AllocaInst, it's not considered as defined
    if (isa<AllocaInst>(V))
        return false;

    // If V is a result of a load from an uninitialized alloca, it's not defined
    if (auto* LI = dyn_cast<LoadInst>(V)) {
        Value* Ptr = LI->getPointerOperand();
        if (auto* AI = dyn_cast<AllocaInst>(Ptr)) {
            // Check if there's any store to this alloca before this load
            bool hasStore = false;
            for (Instruction &PrevInst : BB) {
                if (&PrevInst == &I) 
                    break;
                if (auto* SI = dyn_cast<StoreInst>(&PrevInst)) {
                    if (SI->getPointerOperand() == AI) {
                        hasStore = true;
                        break;
                    }
                }
            }
            if (!hasStore) return false;
        }
    }

    V = getActualValue(V);

    for (Instruction &PrevInst : BB) {
        if (&PrevInst == &I) 
            return false;

        if (isDefinedInInstruction(V, PrevInst))
            return true;

        if (auto* SI = dyn_cast<StoreInst>(&PrevInst)) {
            handleStore(SI);
        }
        else if (auto* LI = dyn_cast<LoadInst>(&PrevInst)) {
            handleLoad(LI);
        }
    }
    return false;
}


    Expression* createExpressionFromInstruction(Instruction& I) {

        if (!I.isBinaryOp()) return nullptr;

        Value* LHS = getActualValue(I.getOperand(0));
        Value* RHS = getActualValue(I.getOperand(1));
        unsigned Opcode = I.getOpcode();

        return new Expression(LHS, RHS, Opcode);

    }


    void buildUniversalSet(Function &F) {
        errs() << "Building Universal Set\n";

        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                Expression* expr = createExpressionFromInstruction(I);
                if (expr) {
                    UniversalSet.insert(*expr);
                    delete expr;
                }
            }
        }
        errs() << "Universal Set Size: " << UniversalSet.size() << "\n";
    }


    void analyzeExpressionsInBasicBlock(BasicBlock &BB) {
        // Clear memory tracking for this block
        MemoryStates.clear();
        LoadedValues.clear();
        ValueOrigins.clear();

        for (Instruction &I : BB) {
            if (auto* SI = dyn_cast<StoreInst>(&I)) {
                handleStore(SI);
                continue;
            }

            if (auto* LI = dyn_cast<LoadInst>(&I)) {
                handleLoad(LI);
                continue;
            }


            Expression* expr = createExpressionFromInstruction(I);
            if (!expr) continue;
            Value* LHS = expr->LHS;
            Value* RHS = expr->RHS;

            bool ifLeftOperandDefined = isDefinedBefore(LHS, I, BB);
            bool ifRightOperandDefined = isDefinedBefore(RHS, I, BB);

            if (!ifLeftOperandDefined && !ifRightOperandDefined) {
                V_USE[&BB].insert(*expr);
                errs() << "Added to V_USE: ";
                expr->dump();
            } else {
                V_DEF[&BB].insert(*expr);
                errs() << "Added to V_DEF: ";
                expr->dump();
            }
            delete expr;
        }

    }


    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {

        errs() << "Function: " << F.getName() << "\n";
        buildUniversalSet(F);

        for (BasicBlock &BB : F) {
            errs() << "Basic Block\n";

            // Initialize sets
            IN[&BB] = UniversalSet;
            OUT[&BB] = ExpressionSet();
            V_USE[&BB] = ExpressionSet();
            V_DEF[&BB] = ExpressionSet();

            // Analyze expressions
            analyzeExpressionsInBasicBlock(BB);

            errs() << "V_USE Size: " << V_USE[&BB].size() << "\n";  
            errs() << "V_DEF Size: " << V_DEF[&BB].size() << "\n";
            errs() << "IN Size: " << IN[&BB].size() << "\n";
            errs() << "OUT Size: " << OUT[&BB].size() << "\n";
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