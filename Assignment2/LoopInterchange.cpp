#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>
#include <limits>
#include <deque>

#define DEBUG_TYPE "affine-loop-interchange"

namespace mlir {
    namespace affine {

        #define GEN_PASS_DEF_AFFINELOOPINTERCHANGE
        #include "mlir/Dialect/Affine/Passes.h.inc"

        namespace {

            struct LoopCost {
                double spatialLocality;
                double temporalLocality;
                double parallelismPotential;
                double synchronizationCost;

                static constexpr double weightOfS = 0.4;
                static constexpr double weightOfT = 0.35;
                static constexpr double weightOfP = 0.15;
                static constexpr double weightOfSync = 0.10;

                double getTotalCost() const {
                    return (weightOfS*spatialLocality + weightOfT*temporalLocality + weightOfP*parallelismPotential - weightOfSync*synchronizationCost);
                }

                void dump(llvm::raw_ostream &os) const {
                    os << "Spatial Locality: " << spatialLocality << "\n";
                    os << "Temporal Locality: " << temporalLocality << "\n";
                    os << "Parallelism Potential: " << parallelismPotential << "\n";
                    os << "Synchronization Cost: " << synchronizationCost << "\n";
                    os << "Total Cost: " << getTotalCost() << "\n";
                }
            };

            struct AccessPattern {

                MemRefAccess access;
                bool isRead;
                Operation *op;

            };

            struct AffineLoopInterchange : public impl::AffineLoopInterchangeBase<AffineLoopInterchange> {
                
                void runOnOperation() override {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Running Affine Loop Interchange Pass\n\n";
                    );
                    
                    func::FuncOp function = getOperation();

                    // walk over all loop nests (outermost loops only)
                    function.walk([&] (AffineForOp outerloop) {
                        
                        if(!isa<func::FuncOp> (outerloop->getParentOp())) {
                            return;
                        }
                        
                        bool isPerfectNest = isPerfectLoopNest(outerloop);
                        if(isPerfectNest) {

                            LLVM_DEBUG(
                                llvm::dbgs() << "Found Perfect Loop Nest\n";
                            );
                            processLoopNest(outerloop);

                        }
                        else {

                            LLVM_DEBUG(
                                llvm::dbgs() << "Found imperfect loop nest\n";
                            );
                            processImperfectNest(outerloop);

                        }

                    });
                }


            private:

                bool isPerfectLoopNest(AffineForOp outerloop) {

                    unsigned int innerLoopCount = 0;
                    unsigned int otherOpCount = 0;

                    for(Operation &operation : outerloop.getBody()->getOperations()) {

                        /* Excluding terminator which is yield */
                        if(isa<AffineYieldOp> (operation)) {
                            continue;
                        }

                        if(AffineForOp innerloop = dyn_cast<AffineForOp>(operation)) {
                            innerLoopCount++;
                        }
                        else otherOpCount++;

                    }

                    return (innerLoopCount == 1 && otherOpCount == 0);
                }

                bool containsIfOperations(AffineForOp loop) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Checking for If Conditionals \n\n";
                    );

                    bool hasIfOperation = false;
                    loop.walk([&] (AffineIfOp ifOperation) {
                        hasIfOperation = true;
                    });

                    return hasIfOperation;
                }

                bool hasRectangularLoopBounds(AffineForOp loop) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Checking for Rectangular Loop Bounds \n\n";
                    );

                    AffineMap lowerBoundMap = loop.getLowerBoundMap();
                    AffineMap upperBoundMap = loop.getUpperBoundMap();

                    /* For rectangular loops, lower and upper bounds should depend only on constants and symbols */

                    for(unsigned int i = 0;i < lowerBoundMap.getNumResults(); i++) {

                        AffineExpr expression = lowerBoundMap.getResult(i);
                        if(expression.isPureAffine() && !expression.isSymbolicOrConstant()) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Lower Bound is not a constant \n";
                            );
                            return false;
                        }

                    }

                    for(unsigned int i = 0;i < upperBoundMap.getNumResults(); i++) {

                        AffineExpr expression = upperBoundMap.getResult(i);
                        if(expression.isPureAffine() && !expression.isSymbolicOrConstant()) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Upper Bound is not a constant \n";
                            );
                            return false;
                        }

                    }

                    // Check nested loop recursively
                    for(Operation &operation : loop.getBody()->getOperations()) {

                        if(AffineForOp nestedLoop = dyn_cast<AffineForOp> (operation)) {
                            if(!hasRectangularLoopBounds(nestedLoop)) {
                                return false;
                            }
                        }

                    }

                    return true;
                }

                bool shouldProcessLoop(AffineForOp loop) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Checking if loop nest should be processed or not \n\n";
                    );

                    // We need to skip the loop with conditionals inside and also the loops with non rectangular bounds
                    if(containsIfOperations(loop)) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "If Conditional Found\n";
                        );
                        return false;
                    }

                    if(!hasRectangularLoopBounds(loop)) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Non Rectangular Loop Found\n";
                        );
                        return false;
                    }


                    return true;
                }

                void collectNestedLoops(AffineForOp outermostLoop, SmallVectorImpl<AffineForOp> &loopNest) {
                    LLVM_DEBUG(llvm::dbgs() << "Collecting Loop Nest starting from outermost loop\n\n");
                    
                    loopNest.clear();
                    loopNest.push_back(outermostLoop);

                    AffineForOp currentLoop = outermostLoop;
                    while(true) {

                        Block& body = currentLoop.getRegion().front();
                        if(body.getOperations().size() != 2) {
                            break;
                        }

                        Operation& firstOperation = *body.begin();
                        if(!isa<AffineForOp>(firstOperation)) {
                            break;
                        }

                        AffineForOp nestedLoop = cast<AffineForOp>(firstOperation);
                        loopNest.push_back(nestedLoop);
                        currentLoop = nestedLoop;
                    }

                    if(loopNest.size() <= 1) {
                        LLVM_DEBUG(llvm::dbgs() << "Not a loop Nest, Skipping this case \n");
                    }
                    else {
                        LLVM_DEBUG(llvm::dbgs() << "Found a valid loop nest, processing this case \n");
                    }

                    return;
                }

                void collectMemoryAccesses(AffineForOp loop, SmallVectorImpl<AccessPattern> &accesses) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Collecting Memory Accesses in the loop nest\n\n";
                    );

                    loop.walk([&] (Operation *operation) {
                        if(auto loadOperation = dyn_cast<AffineLoadOp>(operation)) {
                            accesses.push_back({MemRefAccess(loadOperation), true, operation});
                        }
                        else if(auto storeOperation = dyn_cast<AffineStoreOp>(operation)) {
                            accesses.push_back({MemRefAccess(storeOperation), false, operation});
                        }
                    });

                    return;
                }

                bool performDependenceAnalysis(AffineForOp loop1, AffineForOp loop2) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Running Dependence Analysis\n\n";
                    );

                    /* Collecting all memory accesses in a loop nest */
                    SmallVector<Operation*, 8> memoryAccesses;
                    loop1.walk([&](Operation* operation) {

                        if(isa<AffineStoreOp>(operation) || isa<AffineLoadOp>(operation)) {
                            memoryAccesses.push_back(operation);
                        }

                    });

                    LLVM_DEBUG(
                        llvm::dbgs() << "Found " << memoryAccesses.size() << " memory accesses\n";
                    );

                    /* Interchange is safe if no memory accesses */
                    if(memoryAccesses.empty()) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "No memory accesses \n";
                        );

                        return true;
                    }

                    /* Check for stencil patterns */
                    
                    bool hasLoadWithNegativeOuterIndex = false;
                    bool hasStoreWithCurrentOuterIndex = false;
                    Value OuterInductionVariable = loop1.getInductionVar();

                    for(Operation* operation : memoryAccesses) {

                        if(auto loadOperation = dyn_cast<AffineLoadOp>(operation)) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Found Load Operation: ";
                            );
                            LLVM_DEBUG(
                                loadOperation.dump();
                            );

                            AffineMap loadMap = loadOperation.getAffineMap();
                            ValueRange indices = loadOperation.getMapOperands();

                            /* If Load uses outer IV */
                            for(unsigned int i = 0;i < indices.size(); i++) {

                                if(indices[i] == OuterInductionVariable) {
                                    AffineExpr expression = loadMap.getResult(i);

                                    if(auto binExpression = dyn_cast<AffineBinaryOpExpr>(expression)) {
                                        if(binExpression.getKind() == AffineExprKind::Add) {
                                            if(auto dimExpression = dyn_cast<AffineDimExpr>(binExpression.getLHS())) {
                                                if(dimExpression.getPosition() == i) {
                                                    if(auto constExpression = dyn_cast<AffineConstantExpr>(binExpression.getRHS())) {
                                                        if(constExpression.getValue() < 0) {

                                                            /* Corresponding to patterns like t-K as t+(-K) as AffineExprKind::Sub is not available */
                                                            hasLoadWithNegativeOuterIndex = true;
                                                            LLVM_DEBUG(
                                                                llvm::dbgs() << " Found load with a negative outer index \n";
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                            }
                        }

                        else if (auto storeOperation = dyn_cast<AffineStoreOp>(operation)) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Found Store Operation \n";
                            );
                            LLVM_DEBUG(
                                storeOperation.dump();
                            );

                            AffineMap storeMap = storeOperation.getAffineMap();
                            ValueRange indices = storeOperation.getMapOperands();

                            /* Check is store use outer IV like we did in above if case */
                            for(unsigned int i = 0;i < indices.size(); i++) {

                                if(indices[i] == OuterInductionVariable) {

                                    AffineExpr expression = storeMap.getResult(i);
                                    if(auto dimExpression = dyn_cast<AffineDimExpr>(expression)) {

                                        if(dimExpression.getPosition() == i) {
                                            /* This corresponds to direct use of iv */
                                            hasStoreWithCurrentOuterIndex = true;
                                            LLVM_DEBUG(
                                                llvm::dbgs() << " Found Store with current outer index \n";
                                            );
                                        }

                                    }

                                    else if(auto binExpression = dyn_cast<AffineBinaryOpExpr>(expression)) {

                                        if(binExpression.getKind() == AffineExprKind::Add) {
                                            if(auto dimExpression = dyn_cast<AffineDimExpr>(binExpression.getLHS())) {

                                                if(dimExpression.getPosition() == i) {
                                                    if(auto constExpression = dyn_cast<AffineConstantExpr>(binExpression.getRHS())) {
                                                        if(constExpression.getValue() == 0) {

                                                            /* Found pattern [t,..] */
                                                            hasStoreWithCurrentOuterIndex = true;
                                                            LLVM_DEBUG(
                                                                llvm::dbgs() << "Found store with outer index (t+0) form\n";
                                                            );
                                                        }
                                                    }
                                                }

                                            }
                                        }

                                    }
                                }

                            }

                        }

                    }

                    /* For Stencil Pattern interchange is illegal */
                    if(hasLoadWithNegativeOuterIndex && hasStoreWithCurrentOuterIndex) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Illegal Interchange\n";
                            return false;
                        );
                    }

                    /* If Stencil pattern is not found then do dependence analyis as usual */
                    SmallVector<MemRefAccess, 4> sourceAccesses, destinationAccesses;
                    for(Operation* operation : memoryAccesses) {

                        if(auto storeOperation = dyn_cast<AffineStoreOp>(operation)) {
                            sourceAccesses.push_back(MemRefAccess(storeOperation));
                            destinationAccesses.push_back(MemRefAccess(storeOperation)); /* For WAW */
                        }
                        else if(auto loadOperation = dyn_cast<AffineLoadOp>(operation)) {
                            destinationAccesses.push_back(MemRefAccess(loadOperation));
                        }

                    }

                    LLVM_DEBUG(
                        llvm::dbgs() << "Checking Dependencies between: " << sourceAccesses.size() << " source and " << destinationAccesses.size() << " destination accesses\n";
                    );

                    /* Checking between all pairs of accesses */
                    for(const MemRefAccess &sourceAccess : sourceAccesses) {
                        for(const MemRefAccess &destAccess : destinationAccesses) {

                            /* No dependency when accessing different locations */
                            if(sourceAccess.memref != destAccess.memref) continue;

                            LLVM_DEBUG(
                                llvm::dbgs() << "Analyzing dependence for same memrefs\n";
                            );
                            LLVM_DEBUG(
                                sourceAccess.memref.dump();
                            );

                            FlatAffineValueConstraints dependenceConstraints;
                            SmallVector<DependenceComponent, 2> dependenceComponents;

                            unsigned int loopDepth = 0;
                            Operation* op = loop2.getOperation();

                            while(op) {
                                if(isa<AffineForOp>(op)) {
                                    loopDepth++;
                                }

                                op = op->getParentOp();
                            }

                            loopDepth = std::max(2u,loopDepth);

                            /* I am using MLIR inbuilt functionality for dependence analysis here */
                            DependenceResult dependenceResult = checkMemrefAccessDependence(sourceAccess, destAccess, loopDepth, &dependenceConstraints, &dependenceComponents);

                            if(dependenceResult.value != DependenceResult::NoDependence) {
                                LLVM_DEBUG(
                                    llvm::dbgs() << "Found Dependence between accesses\n";
                                );

                                /* Determine the Type of Dependence */
                                bool isSourceLoadOperation = isa<AffineLoadOp>(sourceAccess.opInst);
                                bool isDestLoadOperation = isa<AffineLoadOp>(destAccess.opInst);

                                bool isFlowDependence = !isSourceLoadOperation && isDestLoadOperation;
                                bool isAntiDependence = isSourceLoadOperation && !isDestLoadOperation;
                                bool isOutputDependence = !isSourceLoadOperation && !isDestLoadOperation;

                                if(isFlowDependence) {
                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Flow Dependence \n";
                                    );
                                }
                                else if(isAntiDependence) {
                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Anti Dependence \n";
                                    );
                                }
                                else if(isOutputDependence) {
                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Output Dependence \n";
                                    );
                                }
                                else {
                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Unknown Dependence Found \n";
                                    );
                                }

                                /* Check for those dependencies that cannot let interchange happen */
                                if(dependenceComponents.size() > 1) {

                                    const DependenceComponent &outercomponent = dependenceComponents[0];
                                    const DependenceComponent &innercomponent = dependenceComponents[1];

                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Outer Loop component bounds: [" << outercomponent.lb << ", " << outercomponent.ub << "]\n";
                                    );
                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Inner Loop component bound: [" << innercomponent.lb << ", " << innercomponent.ub << "]\n";
                                    );

                                    const int LT = -1;
                                    const int EQ = 0;
                                    const int GT = 1;
                                    
                                    int outerDirection;
                                    if(outercomponent.lb > 0) {
                                        outerDirection = GT;
                                    }
                                    else if(outercomponent.ub < 0) {
                                        outerDirection = LT;
                                    }
                                    else if(outercomponent.lb == 0 && outercomponent.ub == 0) {
                                        outerDirection = EQ;
                                    }
                                    

                                    int innerDirection;
                                    if(innercomponent.lb > 0) {
                                        innerDirection = GT;
                                    }
                                    else if(innercomponent.ub < 0) {
                                        innerDirection = LT;
                                    }
                                    else if(innercomponent.lb == 0 && innercomponent.ub == 0) {
                                        innerDirection = EQ;
                                    }

                                    LLVM_DEBUG(
                                        llvm::dbgs() << "Direction Vector before interchange: (" << outerDirection << ", " << innerDirection << ")\n";
                                    );

                                    bool interchangeisLegal = true;

                                    if((outerDirection == EQ) && (innerDirection == EQ)) {
                                        interchangeisLegal = true;
                                        LLVM_DEBUG(
                                            llvm::dbgs() << "Direction Vector indicates no loop-carried dependence, interchange is legal\n";
                                        );
                                    }

                                    else if(innerDirection == LT) {
                                        interchangeisLegal = false;
                                        LLVM_DEBUG(
                                            llvm::dbgs() << "Direction Vector after interchange would be illegal (inner direction is LT), interchange is not allowed\n";
                                        );
                                    }

                                    else if((innerDirection == EQ) && (outerDirection == LT)) {
                                        interchangeisLegal = false;
                                        LLVM_DEBUG(
                                            llvm::dbgs() << "Direction Vector after interchange would be illegal (outer direction is LT and inner is EQ), interchange is not allowed\n";
                                        );
                                    }

                                    if((isFlowDependence) && !(outerDirection == EQ && innerDirection == EQ) && (innerDirection != GT)) {
                                        interchangeisLegal = false;
                                        LLVM_DEBUG(
                                            llvm::dbgs() << "Flow dependence with illegal direction vector after interchange, interchange is not allowed\n";
                                        );
                                    }

                                    if(!interchangeisLegal) {
                                        return false;
                                    }
                                }
                            }
                        }
                    }

                    

                    LLVM_DEBUG(
                        llvm::dbgs() << "Dependence Analysis Passed - interchange is legal\n";
                    );

                    return true;
                }

                bool isLegalInterchange(AffineForOp loop1, AffineForOp loop2) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Checking if this loop interchange legal or not \n\n";
                    );

                    /* We will perform dependence analysis directly since loops reaching at this point are perfectly nested */
                    bool isLegal = performDependenceAnalysis(loop1, loop2);

                    if(!isLegal) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Dependence Analysis shows interchange is illegal\n";
                        );
                    }
                    else {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Dependence Analysis shows that interchange is legal\n";
                        );  
                    }

                    return isLegal;
                }

                bool isLegalFullPermutation(ArrayRef<AffineForOp> originalLoop, ArrayRef<AffineForOp> permutedLoop) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Checking if permutation is valid or not\n\n";
                    );

                    /* Loop Permutation is legal if every pair of adjacent loops which are interchanged 
                    in the permutation can be legally interchanged */

                    /* In this function we will assume that we have perfect nests only because we have defined separate function
                    for imperfect loop nests */

                    for(unsigned int i = 0;i < originalLoop.size()-1; i++) {

                        if(!isLegalInterchange(originalLoop[i], originalLoop[i+1])) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Interchange Not Legal between loops at position: " << i << " and " << i+1 << "\n";
                            );

                            return false;
                        }

                    }

                    LLVM_DEBUG(
                        llvm::dbgs() << "Full Permutation is Legal\n";
                    );

                    return true;
                }

                void findLegalPermutations(ArrayRef<AffineForOp> loopNest, SmallVectorImpl<SmallVector<AffineForOp,4>>& legalPermutations) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Finding legal permutations of the loop nest\n\n";
                    );

                    /* Original Order is a legal one */
                    legalPermutations.push_back(SmallVector<AffineForOp, 4>(loopNest.begin(), loopNest.end()));
                    
                    /* Indices Vector to generate different permutations using std */
                    SmallVector<AffineForOp, 4> loopPermutation(loopNest.begin(), loopNest.end());
                    SmallVector<unsigned int, 4> indices;

                    for(unsigned int i = 0; i < loopNest.size(); i++) {
                        indices.push_back(i);
                    }

                    /* Generating and processing different permutations */
                    do {

                        /* Create the permutation */
                        for(unsigned int i = 0; i < indices.size(); i++) {
                            loopPermutation[i] = loopNest[indices[i]];
                        }

                        /* Check if this is a new, legal permutation */
                        bool isNew = true;
                        for(const auto &existingPermutation : legalPermutations) {
                            if(std::equal(existingPermutation.begin(), existingPermutation.end(), loopPermutation.begin(), loopPermutation.end())) {
                                isNew = false;
                                break;
                            }
                        }

                        if(isNew && isLegalFullPermutation(loopNest, loopPermutation)) {
                            legalPermutations.push_back(loopPermutation);
                        }

                    } while (std::next_permutation(indices.begin(), indices.end()));

                    return;
                }

                bool isIVUsedInDimension(AffineExpr expression, unsigned int position) {

                    if(auto dimensionExpression = dyn_cast<AffineDimExpr> (expression)) {
                        return dimensionExpression.getPosition() == position;
                    } 
                    else if(AffineBinaryOpExpr binOp = dyn_cast<AffineBinaryOpExpr> (expression)) {

                        if(binOp.getKind() == AffineExprKind::Mod) {
                            return isIVUsedInDimension(binOp.getLHS(), position);
                        }
                        else if(binOp.getKind() == AffineExprKind::CeilDiv) {
                            return isIVUsedInDimension(binOp.getLHS(), position);
                        }

                        return isIVUsedInDimension(binOp.getLHS(), position) || isIVUsedInDimension(binOp.getRHS(), position);
                    }
                    
                    return false;
                }

                double analyzeSpatialLocality(ArrayRef<AffineForOp> loopNest, ArrayRef<AccessPattern> accesses) {
                    if(loopNest.size() == 0 || accesses.size() == 0) {
                        return 0.0;
                    }

                    double spatialscore = 0.0;
                    double totalweight = 0.0;

                    AffineForOp innermostloop = loopNest.back();
                    Value innermostIV = innermostloop.getInductionVar();

                    for(const AccessPattern &access : accesses) {
                        MemRefType memreftype = cast<MemRefType>(access.access.memref.getType());
                        int numDimensions = memreftype.getRank();

                        if(numDimensions < 2) {
                            continue;
                        }

                        AffineMap accessMap;
                        SmallVector<Value> mapOperands;

                        if(AffineLoadOp loadOperation = dyn_cast<AffineLoadOp> (access.op)) {
                            accessMap = loadOperation.getAffineMap();
                            mapOperands.assign(loadOperation.getMapOperands().begin(), loadOperation.getMapOperands().end());
                        }
                        else if(AffineStoreOp storeOperation = dyn_cast<AffineStoreOp> (access.op)) {
                            accessMap = storeOperation.getAffineMap();
                            mapOperands.assign(storeOperation.getMapOperands().begin(), storeOperation.getMapOperands().end());
                        }
                        else {
                            continue;
                        }

                        /* For each access compute its score */
                        double access_score = 0.0;
                        double access_weight;

                        if(access.isRead) {
                            access_weight = 1.0;
                        }
                        else access_weight = 2.0;

                        /* Look for fastest varying dimension */
                        if(accessMap.getNumResults() > 0) {

                            unsigned int fastestvaryingDim = accessMap.getNumResults()-1;
                            AffineExpr fastestdimExpr = accessMap.getResult(fastestvaryingDim);

                            bool innermostIVinfastestDim = false;

                            for(unsigned int i = 0;i < mapOperands.size(); i++) {

                                if(mapOperands[i] == innermostIV) {
                                    if(isIVUsedInDimension(fastestdimExpr,i)) {
                                        innermostIVinfastestDim = true;
                                        break;
                                    }
                                }

                            }

                            if(innermostIVinfastestDim) {
                                
                                if(access.isRead) {
                                    access_score = 1;
                                }
                                else access_score = 1.3;

                            }
                            else {

                                bool innermostInSecondFastest = false;
                                if(accessMap.getNumResults() >= 2) {

                                    unsigned int secondFastestDimension = accessMap.getNumResults()-2;
                                    AffineExpr secondDimExpr = accessMap.getResult(secondFastestDimension);

                                    for(unsigned int i = 0;i < mapOperands.size(); i++) {
                                        
                                        if(mapOperands[i] == innermostIV && isIVUsedInDimension(secondDimExpr,i)) {
                                            innermostInSecondFastest = true;
                                            break;
                                        }

                                    }

                                } 

                                if(innermostInSecondFastest) {
                                    access_score = 0.4;
                                }
                                else access_score = 0.1;

                            }

                        }

                        spatialscore += access_score * access_weight;
                        totalweight += access_weight;
                    }

                    /* make sure floats are there */
                    return totalweight > 0.0 ? spatialscore / totalweight : 0.0;
                }

                bool isIVUsedInExpression(AffineExpr expression, Value IV, AffineMap map, ArrayRef<Value> mapOperands) {

                    if(AffineDimExpr dimExpression = dyn_cast<AffineDimExpr>(expression)) {

                        unsigned int dimposition = dimExpression.getPosition();
                        if(dimposition < mapOperands.size()) {
                            return mapOperands[dimposition] == IV;
                        }
                        return false;

                    }
                    else if(AffineBinaryOpExpr binOp = dyn_cast<AffineBinaryOpExpr> (expression)) {

                        return isIVUsedInExpression(binOp.getLHS(), IV, map, mapOperands) || isIVUsedInExpression(binOp.getRHS(), IV, map, mapOperands);

                    }

                    return false;
                }

                double analyzeTemporalLocality(ArrayRef<AffineForOp> loopNest, ArrayRef<AccessPattern> accesses) {
                    if(loopNest.size() == 0 || accesses.size() == 0) {
                        return 0.0;
                    }

                    double temporalScore = 0.0;
                    double totalWeight = 0.0;

                    /* Group Accesses record by memref */
                    DenseMap<Value, SmallVector<const AccessPattern*, 4>> memRefToAccesses;
                    for(const AccessPattern &access : accesses) {
                        memRefToAccesses[access.access.memref].push_back(&access);
                    }

                    /* Analyze Temporal Locality */
                    for(const auto &entry : memRefToAccesses) {
                        
                        const auto &accessList = entry.second;

                        bool isOutput = false;
                        for(const auto *access : accessList) {
                            
                            if(!access->isRead) {
                                isOutput = true;
                                break;
                            }

                        }

                        DenseMap<unsigned int, SmallVector<unsigned int, 4>> dimToLoopDepth;
                        for(const auto *access : accessList) {

                            AffineMap accessMap;
                            SmallVector<Value> mapOperands;

                            if(AffineLoadOp loadOperation = dyn_cast<AffineLoadOp>(access->op)) {
                                accessMap = loadOperation.getAffineMap();
                                mapOperands.assign(loadOperation.getMapOperands().begin(), loadOperation.getMapOperands().end());
                            }
                            else if(AffineStoreOp storeOperation = dyn_cast<AffineStoreOp>(access->op)) {
                                accessMap = storeOperation.getAffineMap();
                                mapOperands.assign(storeOperation.getMapOperands().begin(), storeOperation.getMapOperands().end());
                            }
                            else continue;


                            for(unsigned int dimensionindex = 0; dimensionindex < accessMap.getNumResults(); dimensionindex++) {

                                AffineExpr dimensionExpr = accessMap.getResult(dimensionindex);
                                int innermostDepth = -1;

                                for(unsigned int i = 0;i < loopNest.size(); i++) {

                                    Value IV = const_cast<AffineForOp &>(loopNest[i]).getInductionVar();
                                    if(isIVUsedInExpression(dimensionExpr, IV, accessMap, mapOperands)) {
                                        innermostDepth = i;
                                    }

                                }

                                if (innermostDepth != -1) {
                                    dimToLoopDepth[dimensionindex].push_back(innermostDepth);
                                }

                            }
                        }


                        double memrefScore = 0.0;
                        double memrefWeight;

                        if(isOutput) {
                            memrefWeight = 2.0;
                        }
                        else memrefWeight = 1.0;

                        for(const auto& dimEntry : dimToLoopDepth) {

                            const auto& depths = dimEntry.second;

                            if(depths.size() == 0) {
                                memrefScore += 1.0;
                                continue;
                            }

                            double averageDepth = 0.0;
                            for(unsigned int depth : depths) {
                                averageDepth += depth;
                            }

                            averageDepth /= depths.size();
                            double normalizedScore = 1.0 - (averageDepth/loopNest.size());

                            memrefScore += normalizedScore;
                        }

                        memrefScore /= std::max(1u, (unsigned int)dimToLoopDepth.size());
                        temporalScore += memrefScore * memrefWeight;
                        totalWeight += memrefWeight;
                    }

                    return totalWeight > 0.0 ? temporalScore / totalWeight : 0.0;
                }



                double analyzeParallelismChance(ArrayRef<AffineForOp> loopNest) {
                    if(loopNest.size() == 0) {
                        return 0.0;
                    }

                    double parallelism = 0.0;
                    for(unsigned int i = 0; i < loopNest.size(); i++) {

                        AffineForOp loop = loopNest[i];
                        bool isParallelizable;

                        if(loop.getStep() != 1) return false;
                        else isParallelizable = isLoopParallel(loop);

                        if(isParallelizable) {
                            double positionScore = 1 - (static_cast<double> (i) / loopNest.size());
                            parallelism += positionScore;
                        }

                    }

                    return parallelism;
                }

                double analyzeSyncCost(ArrayRef<AffineForOp> loopNest) {
                    if(loopNest.size() == 0) return 0;

                    double SyncCost = 0.0;
                    bool foundSequential = false;

                    for(unsigned int i = 0;i < loopNest.size(); i++) {

                        AffineForOp loop = loopNest[i];
                        bool isParallel;

                        if (loop.getStep() != 1) return false;
                        else isParallel = isLoopParallel(loop);

                        if(isParallel && foundSequential) {
                            
                            double depthFactor = static_cast<double> (i) / loopNest.size();
                            SyncCost += depthFactor;

                        }

                        if(!isParallel) {
                            foundSequential = true;
                        }

                    }

                    return SyncCost;
                }

                LoopCost analyzeCost(ArrayRef<AffineForOp> loopNest, ArrayRef<AccessPattern> accesses) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Analyzing loop cost\n";
                    );

                    LoopCost cost;
                    cost.spatialLocality = analyzeSpatialLocality(loopNest,accesses);
                    cost.temporalLocality = analyzeTemporalLocality(loopNest, accesses);
                    cost.parallelismPotential = analyzeParallelismChance(loopNest);
                    cost.synchronizationCost = analyzeSyncCost(loopNest);
                    
                    LLVM_DEBUG(
                        cost.dump(llvm::dbgs());
                    );
                    return cost;
                }

                void applyPermutation(ArrayRef<AffineForOp> originalLoops, ArrayRef<AffineForOp> permutedLoops) {
                    if(originalLoops.size() != permutedLoops.size() || originalLoops.size() == 0) {
                        return;
                    }
                
                    /* Simple Case for 2 loops */
                    if(originalLoops.size() == 2) {
                        interchangeLoops(originalLoops[0], permutedLoops[0]);
                        return;
                    }
                
                    /* > 2 loop nest */
                    SmallVector<AffineForOp, 4> currentOrder(originalLoops.begin(), originalLoops.end());
                    bool changed = true;
                
                    while(changed) {
                        changed = false;
                        for(unsigned int i = 0;i < currentOrder.size()-1; i++) {
                            unsigned currentindex = 0, nextindex = 0;
                            
                            for(unsigned int j = 0;j < permutedLoops.size(); j++) {
                                if(permutedLoops[j] == currentOrder[i]) {
                                    currentindex = j;
                                }
                                if(permutedLoops[j] == currentOrder[i+1]) {
                                    nextindex = j;
                                }
                            }
                            
                            if(currentindex > nextindex) {

                                interchangeLoops(currentOrder[i], currentOrder[i+1]);
                                std::swap(currentOrder[i], currentOrder[i+1]);
                                changed=true;

                            }
                        }
                    }
                
                    return;
                }

                /* Function Overloading */
                // LogicalResult interchangeLoops(AffineForOp loop1, AffineForOp loop2) {
                //     LLVM_DEBUG(
                //         llvm::dbgs() << "Performing loop interchange\n";
                //     );

                //     mlir::affine::interchangeLoops(loop1,loop2);
                //     return success();
                // }

                void handleImperfectNest(AffineForOp outerLoop) {
                    LLVM_DEBUG(llvm::dbgs() << "Handling imperfect nest\n");
                    
                    /* CREDITS: This function has been taken from ChatGPT to perform interchange in case of imperfect */
                    OpBuilder builder(outerLoop);
                    Location loc = outerLoop.getLoc();
                    
                    // Get all inner loops
                    SmallVector<AffineForOp, 4> innerLoops;
                    for (Operation &op : outerLoop.getBody()->without_terminator()) {
                      if (auto forOp = dyn_cast<AffineForOp>(op))
                        innerLoops.push_back(forOp);
                    }
                    
                    // We need to collect information about each inner loop
                    struct InnerLoopInfo {
                      int64_t lowerBound;
                      int64_t upperBound;
                      Value storeValue;
                      Value memref;
                    };
                    
                    SmallVector<InnerLoopInfo, 4> loopInfos;
                    
                    // Collect information about each inner loop
                    for (AffineForOp innerLoop : innerLoops) {
                      InnerLoopInfo info;
                      info.lowerBound = innerLoop.getConstantLowerBound();
                      info.upperBound = innerLoop.getConstantUpperBound();
                      
                      // Find the memref and value being stored
                      // Assumes a single store operation in each inner loop
                      innerLoop.walk([&](AffineStoreOp storeOp) {
                        info.storeValue = storeOp.getValue();
                        info.memref = storeOp.getMemRef();
                      });
                      
                      loopInfos.push_back(info);
                    }
                    
                    // Set insertion point after the original loop
                    builder.setInsertionPointAfter(outerLoop);
                    
                    // Create interchanged loops for each inner loop
                    for (const InnerLoopInfo &info : loopInfos) {
                      // Create new outer loop (was inner loop)
                      auto newOuterLoop = builder.create<AffineForOp>(
                          loc, info.lowerBound, info.upperBound);
                      
                      // Create new inner loop (was outer loop)
                      builder.setInsertionPointToStart(newOuterLoop.getBody());
                      auto newInnerLoop = builder.create<AffineForOp>(
                          loc, 0, outerLoop.getConstantUpperBound());
                      
                      // Create the store operation with swapped indices
                      builder.setInsertionPointToStart(newInnerLoop.getBody());
                      
                      // Create new map for interchanged indices
                      SmallVector<AffineExpr, 2> indices;
                      indices.push_back(builder.getAffineDimExpr(0)); // First index (j)
                      indices.push_back(builder.getAffineDimExpr(1)); // Second index (i)
                      
                      auto newMap = AffineMap::get(2, 0, indices, builder.getContext());
                      
                      // Create the new store with interchanged indices
                      builder.create<AffineStoreOp>(
                          loc,
                          info.storeValue,
                          info.memref,
                          newMap,
                          ValueRange{newOuterLoop.getInductionVar(), newInnerLoop.getInductionVar()});
                      
                      // Move insertion point for next outer loop
                      builder.setInsertionPointAfter(newOuterLoop);
                    }
                    
                    // Erase the original loop nest
                    outerLoop.erase();
                    
                    LLVM_DEBUG(llvm::dbgs() << "Successfully interchanged imperfect nest\n");
                  }

                void processImperfectNest(AffineForOp outerLoop) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "Processing imperfect loop nest\n";
                    );
                    
                    
                    SmallVector<AffineForOp, 4> innerLoops;
                    
                    for (Operation &operation : outerLoop.getBody()->getOperations()) {
                        /* Skip terminator which is yeild */
                        if (isa<AffineYieldOp>(operation))
                            continue;
                            
                        if (auto innerLoop = dyn_cast<AffineForOp>(operation)) {
                            innerLoops.push_back(innerLoop);
                        }
                    }
                    
                    LLVM_DEBUG(llvm::dbgs() << "Found " << innerLoops.size() << " inner loops in imperfect nest\n");
                    
                    /*and have simple access patterns that would benefit from interchange*/
                    Value commonMemref = nullptr;
                    bool columnMajorAccess = true;  
                    
                    for (AffineForOp innerLoop : innerLoops) {
                    
                        if (!performDependenceAnalysis(outerLoop, innerLoop)) {
                            LLVM_DEBUG(llvm::dbgs() << "Interchange not legal for at least one loop pair\n");
                            return;  
                        }
                        
                        // Check access pattern
                        innerLoop.walk([&](AffineStoreOp storeOp) {
                            Value memref = storeOp.getMemRef();
                            if (!commonMemref) {
                                commonMemref = memref;
                            } else if (commonMemref != memref) {
                                columnMajorAccess = false;  /* If it is not column major then this will mark it false */
                            }
                            
                            
                            AffineMap map = storeOp.getAffineMap();
                            if (map.getNumResults() == 2) {  
                                SmallVector<Value> mapOperands(storeOp.getMapOperands());
                                Value innerIV = innerLoop.getInductionVar();
                                Value outerIV = outerLoop.getInductionVar();
                                
                                bool innerInFirstDim = false;
                                bool outerInSecondDim = false;
                                
                                for (unsigned i = 0; i < mapOperands.size(); i++) {
                                    if (mapOperands[i] == innerIV && 
                                        isIVUsedInDimension(map.getResult(0), i)) {
                                        innerInFirstDim = true;
                                    }
                                    if (mapOperands[i] == outerIV && 
                                        isIVUsedInDimension(map.getResult(1), i)) {
                                        outerInSecondDim = true;
                                    }
                                }
                                
                                if (!(innerInFirstDim && outerInSecondDim)) {
                                    columnMajorAccess = false;  /* Not Column Major */
                                }
                            } else {
                                columnMajorAccess = false;  
                            }
                        });
                    }
                    
                    /* Different Cases */
                    if (columnMajorAccess && commonMemref) {
                        LLVM_DEBUG(llvm::dbgs() << "Detected column-major access pattern, proceeding with interchange\n");
                        handleImperfectNest(outerLoop);
                        return;
                    }
                    
                    
                    for (AffineForOp innerLoop : innerLoops) {
                        LLVM_DEBUG(llvm::dbgs() << "Processing 2-level nest within imperfect structure\n");
                        
                    
                        if (!performDependenceAnalysis(outerLoop, innerLoop)) {
                            LLVM_DEBUG(llvm::dbgs() << "Interchange not legal for this loop pair\n");
                            continue;
                        }
                        
                        SmallVector<AccessPattern, 8> accesses;
                        collectMemoryAccesses(outerLoop, accesses);
                        
                        SmallVector<AffineForOp, 2> currentNest = {outerLoop, innerLoop};
                        LoopCost originalCost = analyzeCost(currentNest, accesses);
                        
                        LLVM_DEBUG(
                            llvm::dbgs() << "Original two-level nest cost:\n";
                            originalCost.dump(llvm::dbgs());
                        );
                        
                        /* manual cost computations */
                        double spatialLocalityInterchanged = 0.0;
                        double temporalLocalityInterchanged = 0.0;
                        double parallelismPotentialInterchanged = 0.0;
                        double synchronizationCostInterchanged = 0.0;
                        
                        
                        bool isColumnMajor = false;
                        
                        for (const AccessPattern &access : accesses) {
                            MemRefType memrefType = cast<MemRefType>(access.access.memref.getType());
                            int numDimensions = memrefType.getRank();
                            
                            if (numDimensions < 2) continue;
                            
                            AffineMap accessMap;
                            SmallVector<Value> mapOperands;
                            
                            if (AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(access.op)) {
                                accessMap = loadOp.getAffineMap();
                                mapOperands.assign(loadOp.getMapOperands().begin(), loadOp.getMapOperands().end());
                            } else if (AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(access.op)) {
                                accessMap = storeOp.getAffineMap();
                                mapOperands.assign(storeOp.getMapOperands().begin(), storeOp.getMapOperands().end());
                            } else {
                                continue;
                            }
                            
                            Value innerIV = innerLoop.getInductionVar();
                            Value outerIV = outerLoop.getInductionVar();
                            
                            if (accessMap.getNumResults() == 2) {
                                for (unsigned i = 0; i < mapOperands.size(); i++) {
                                    if (mapOperands[i] == innerIV && 
                                        isIVUsedInDimension(accessMap.getResult(0), i)) {
                                        isColumnMajor = true;
                                    }
                                }
                            }
                            
                            
                            spatialLocalityInterchanged += isIVUsedInFastestDimension(accessMap, mapOperands, innerIV) ? 1.0 : 0.2;
                            temporalLocalityInterchanged += isIVUsedInFastestDimension(accessMap, mapOperands, outerIV) ? 0.5 : 1.0;
                        }
                        
                        spatialLocalityInterchanged /= std::max(1, (int)accesses.size());
                        temporalLocalityInterchanged /= std::max(1, (int)accesses.size());
                        
                        parallelismPotentialInterchanged = isLoopParallel(innerLoop) ? 1.0 : 0.0; 
                        synchronizationCostInterchanged = isLoopParallel(outerLoop) ? 0.5 : 0.0;
                        
                        /*Construct interchanged cost*/
                        LoopCost interchangedCost;
                        interchangedCost.spatialLocality = spatialLocalityInterchanged;
                        interchangedCost.temporalLocality = temporalLocalityInterchanged;
                        interchangedCost.parallelismPotential = parallelismPotentialInterchanged;
                        interchangedCost.synchronizationCost = synchronizationCostInterchanged;
                        
                        LLVM_DEBUG(
                            llvm::dbgs() << "Estimated interchanged nest cost:\n";
                            interchangedCost.dump(llvm::dbgs());
                            llvm::dbgs() << "Original total cost: " << originalCost.getTotalCost() << "\n";
                            llvm::dbgs() << "Interchanged total cost: " << interchangedCost.getTotalCost() << "\n";
                        );
                        
                        bool forceInterchange = isColumnMajor;
                        if (forceInterchange) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Special case: Detected column-major access, forcing interchange\n";
                            );
                        }
                        
                        if (interchangedCost.getTotalCost() > originalCost.getTotalCost() || forceInterchange) {
                            LLVM_DEBUG(
                                llvm::dbgs() << "Interchanging loops in imperfect nest\n";
                            );
                            handleImperfectNest(outerLoop);
                            return;  
                        } else {
                            LLVM_DEBUG(llvm::dbgs() << "No benefit to interchanging these loops\n");
                        }
                    }
                }
                
                
                bool isIVUsedInFastestDimension(AffineMap map, ArrayRef<Value> mapOperands, Value iv) {
                    if (map.getNumResults() == 0) return false;
                    
                    unsigned fastestDim = map.getNumResults() - 1;
                    AffineExpr fastestExpr = map.getResult(fastestDim);
                    
                    for (unsigned i = 0; i < mapOperands.size(); i++) {
                        if (mapOperands[i] == iv && isIVUsedInDimension(fastestExpr, i)) {
                            return true;
                        }
                    }
                    
                    return false;
                }
                
                void processLoopNest(AffineForOp outerloop) {
                    LLVM_DEBUG(llvm::dbgs() << "Processing the loop nest\n");

                    /* Collect Loop Nest */
                    SmallVector<AffineForOp, 4> loopNest;
                    collectNestedLoops(outerloop, loopNest);

                    /* Collect Memoruy Access Patterns */
                    SmallVector<AccessPattern, 8> accesses;
                    collectMemoryAccesses(outerloop, accesses);

                    /* Find all legal permutations of loop nest */
                    SmallVector<SmallVector<AffineForOp, 4>, 4> legalPermutations;
                    findLegalPermutations(loopNest, legalPermutations);

                    if(legalPermutations.size() <= 1) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "No legal permutations found, keeping original order\n";
                        );
                        return;
                    }

                    /* Cost analysis for legal permutations */
                    SmallVector<LoopCost, 4> permutationCosts;

                    /* Cost for original loop nest order */
                    LoopCost originalCost = analyzeCost(loopNest,accesses);
                    LLVM_DEBUG(
                        llvm::dbgs() << "Analyzing loop cost\n";
                    );
                    LLVM_DEBUG(
                        originalCost.dump(llvm::dbgs());
                    );
                    permutationCosts.push_back(originalCost);

                    /* Costs for all Legal Perms */
                    for(unsigned int i = 1; i < legalPermutations.size(); i++) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Analyzing Cost for Permutation: " << i << "\n";
                        );
                        LoopCost cost = analyzeCost(legalPermutations[i], accesses);
                        LLVM_DEBUG(
                            cost.dump(llvm::dbgs());
                        );

                        permutationCosts.push_back(cost);
                    }

                    /* Find the permutation with highest cost 
                    (In our case according to analyze spatial and temporal locality function higher cost means better ) */

                    unsigned int bestIndex = 0;
                    double bestCost =permutationCosts[0].getTotalCost();

                    for(unsigned int i = 1;i < permutationCosts.size(); i++) {

                        double currentCost = permutationCosts[i].getTotalCost();
                        LLVM_DEBUG(
                            llvm::dbgs() << "Permutation: " << i << " cost: " << currentCost << "\n";
                        );

                        if(currentCost > bestCost) {
                            
                            bestCost = currentCost;
                            bestIndex = i;

                        }
                    }

                    LLVM_DEBUG(
                        llvm::dbgs() << "Selected Best Permutation: " << bestIndex << " with cost: " << bestCost << "\n";
                    );

                    /* Apply the permutation (loop interchange) if best perm is not the original order */
                    if(bestIndex != 0) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Applying Permutation with index: " << bestIndex << "\n";
                        );
                        applyPermutation(loopNest, legalPermutations[bestIndex]);
                    }
                    else {
                        LLVM_DEBUG(
                            llvm::dbgs() << "Best Permutation is original order only !\n";
                        );
                    }

                    return;
                }
            };

            
        }
        
        std::unique_ptr<OperationPass<func::FuncOp>> createAffineLoopInterchangePass() {
            return std::make_unique<AffineLoopInterchange>();
        }

    }
}