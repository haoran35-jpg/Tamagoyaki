//===- ShloPasses.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Shlo/Shlo.h"

#include "EmatchUtils.h"
#include "EquivalenceUtils.h"
#include "Utils/ClassOpUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>

using namespace mlir;

namespace mlir::shlo {
#define GEN_PASS_DEF_SHLOSATURATEPASS
#define GEN_PASS_DEF_SHLOANNOTATECOSTSPASS
#include "ShloPasses.h.inc"
} // namespace mlir::shlo

namespace {

//===----------------------------------------------------------------------===//
// Native rewrites: each builds exactly one op, so that the `ematch.dedup` the
// rule lowering emits after every native rewrite hash-conses all of them.
//===----------------------------------------------------------------------===//

Operation *buildZeroScalar(PatternRewriter &rewriter, Type tensorType) {
  Type elemTy = cast<ShapedType>(tensorType).getElementType();
  return arith::ConstantOp::create(rewriter, rewriter.getUnknownLoc(),
                                   rewriter.getZeroAttr(elemTy));
}

Operation *buildFill(PatternRewriter &rewriter, Value value, Value init) {
  return linalg::FillOp::create(rewriter, rewriter.getUnknownLoc(),
                                ValueRange{value}, ValueRange{init});
}

Operation *buildMatmul(PatternRewriter &rewriter, Value lhs, Value rhs,
                       Value init) {
  return linalg::MatmulOp::create(rewriter, rewriter.getUnknownLoc(),
                                  ValueRange{lhs, rhs}, ValueRange{init});
}

Operation *buildAdd(PatternRewriter &rewriter, Value lhs, Value rhs,
                    Value init) {
  return linalg::AddOp::create(rewriter, rewriter.getUnknownLoc(),
                               ValueRange{lhs, rhs}, ValueRange{init});
}

Operation *buildTranspose(PatternRewriter &rewriter, Value input, Value init,
                          Attribute permutation) {
  return linalg::TransposeOp::create(rewriter, rewriter.getUnknownLoc(), input,
                                     init,
                                     cast<DenseI64ArrayAttr>(permutation));
}

/// Result type of the matrix product `lhs` x `rhs` (contracting [1] x [0]).
Type dotResultType(PatternRewriter &, Value lhs, Value rhs) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  return RankedTensorType::get({lhsType.getDimSize(0), rhsType.getDimSize(1)},
                               lhsType.getElementType());
}

/// Lower `op` with the upstream stablehlo-legalize-to-linalg, run on a copy of
/// it in a scratch module, and clone the result next to `op`. Unlike the
/// builders above this creates several ops, so it hash-conses each one itself.
/// Returns the values that replace the results of `op`.
LogicalResult lowerViaUpstream(PatternRewriter &rewriter,
                               PDLResultList &results,
                               ArrayRef<PDLValue> args) {
  Operation *op = args[0].cast<Operation *>();
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();

  OwningOpRef<ModuleOp> scratch = ModuleOp::create(loc);
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(scratch->getBody());
  auto func = func::FuncOp::create(
      builder, loc, "oracle",
      builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  IRMapping toScratch;
  toScratch.map(op->getOperands(), entry->getArguments());
  Operation *copy = builder.clone(*op, toScratch);
  // Cost annotations must not leak into the lowering.
  copy->setDiscardableAttrs(DictionaryAttr::get(ctx));
  func::ReturnOp::create(builder, loc, copy->getResults());

  PassManager pm(ctx);
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloLegalizeToLinalgPass());
  // The saturation rewriter cannot recover from a failed rewrite.
  if (failed(pm.run(*scratch)))
    llvm::report_fatal_error(llvm::Twine("LowerViaUpstream: cannot lower ") +
                             op->getName().getStringRef());

  Block &body = func.getBody().front();
  IRMapping toGraph;
  toGraph.map(body.getArguments(), op->getOperands());
  for (Operation &inner : body.without_terminator()) {
    Operation *kept =
        ematch::dedupOperation(rewriter, rewriter.clone(inner, toGraph));
    for (auto [from, to] : llvm::zip(inner.getResults(), kept->getResults()))
      toGraph.map(from, ematch::getClassResult(rewriter, to));
  }

  SmallVector<Value> replacements;
  for (Value v : body.getTerminator()->getOperands())
    replacements.push_back(toGraph.lookupOrDefault(v));
  results.push_back(ValueRange(replacements));
  return success();
}

void registerNativeRewrites(PDLPatternModule &patterns) {
  patterns.registerRewriteFunction("DotResultType", dotResultType);
  patterns.registerRewriteFunction("BuildZeroScalar", buildZeroScalar);
  patterns.registerRewriteFunction("BuildFill", buildFill);
  patterns.registerRewriteFunction("BuildMatmul", buildMatmul);
  patterns.registerRewriteFunction("BuildAdd", buildAdd);
  patterns.registerRewriteFunction("BuildTranspose", buildTranspose);
  patterns.registerRewriteFunction("LowerViaUpstream", lowerViaUpstream);
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

struct ShloSaturatePass
    : public shlo::impl::ShloSaturatePassBase<ShloSaturatePass> {
  using ShloSaturatePassBase::ShloSaturatePassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    ShloSaturatePassBase::getDependentDialects(registry);
    // LowerViaUpstream runs this pass and clones its output into the IR.
    stablehlo::createStablehloLegalizeToLinalgPass()->getDependentDialects(
        registry);
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    OwningOpRef<ModuleOp> patternsModule =
        parseSourceFile<ModuleOp>(patternsFile, module.getContext());
    if (!patternsModule) {
      emitError(module.getLoc())
          << "failed to parse patterns file: " << patternsFile;
      return signalPassFailure();
    }

    ematch::convertEmatchOpsToApplyRewrites(*patternsModule);
    PDLPatternModule pdlPatterns(patternsModule.release());
    registerNativeRewrites(pdlPatterns);

    if (failed(equivalence::restoreClassInvariants(module)))
      return signalPassFailure();

    ematch::runSaturation(module.getContext(), std::move(pdlPatterns), module,
                          maxIters, maxNodes);
  }
};

int64_t numElements(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 1;
  return shaped.getNumElements();
}

/// Number of scalar ops in the single-block body of `region`, at least 1.
int64_t bodyOps(Region &region) {
  if (region.empty())
    return 1;
  int64_t count = 0;
  for (Operation &op : region.front().without_terminator()) {
    (void)op;
    ++count;
  }
  return std::max<int64_t>(count, 1);
}

/// Estimated scalar work of `op` on static shapes (see ShloAnnotateCostsPass).
int64_t estimateWork(Operation *op) {
  if (isa<tensor::EmptyOp, arith::ConstantOp, stablehlo::ConstantOp>(op))
    return 0;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    int64_t points = 1;
    for (int64_t range : linalgOp.getStaticLoopRanges())
      points *= ShapedType::isDynamic(range) ? 1 : range;
    return points * bodyOps(op->getRegion(0));
  }
  if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    auto lhsType = cast<ShapedType>(dot.getLhs().getType());
    int64_t contracted = 1;
    for (int64_t dim :
         dot.getDotDimensionNumbers().getLhsContractingDimensions())
      contracted *= lhsType.isDynamicDim(dim) ? 1 : lhsType.getDimSize(dim);
    return 2 * numElements(dot.getType()) * contracted;
  }
  if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op))
    return numElements(reduce.getInputs().front().getType()) *
           bodyOps(reduce.getBody());
  int64_t elements = 0;
  for (Type type : op->getResultTypes())
    elements += numElements(type);
  return elements;
}

struct ShloAnnotateCostsPass
    : public shlo::impl::ShloAnnotateCostsPassBase<ShloAnnotateCostsPass> {
  using ShloAnnotateCostsPassBase::ShloAnnotateCostsPassBase;

  void runOnOperation() final {
    if (model != "level" && model != "flops") {
      emitError(getOperation().getLoc()) << "unknown cost model: " << model;
      return signalPassFailure();
    }
    bool flops = model == "flops";
    MLIRContext *ctx = &getContext();
    getOperation().walk([&](equivalence::GraphOp graph) {
      for (Operation &op : graph.getBody().front()) {
        if (isa<equivalence::ClassOp, equivalence::YieldOp,
                equivalence::GraphOp>(op))
          continue;
        int64_t cost = isa<stablehlo::StablehloDialect>(op.getDialect())
                           ? stablehloCost
                           : defaultCost;
        if (flops)
          cost += estimateWork(&op);
        op.setAttr("equivalence.cost",
                   equivalence::CostAttr::get(ctx, cost));
      }
    });
  }
};

struct ShloExtractOptions : public PassPipelineOptions<ShloExtractOptions> {
  Option<std::string> prefer{
      *this, "prefer",
      llvm::cl::desc("Dialect to extract when the work estimates tie: "
                     "`linalg`, `stablehlo` or `none`."),
      llvm::cl::init("linalg")};
};

} // namespace

void mlir::shlo::registerShloPipelines() {
  PassPipelineRegistration<ShloExtractOptions>(
      "shlo-extract",
      "Annotate FLOP costs, select greedily and extract one program from "
      "every equivalence.graph",
      [](OpPassManager &pm, const ShloExtractOptions &options) {
        // A per-op penalty far above any work estimate of the other side.
        constexpr int64_t penalty = int64_t(1) << 40;
        ShloAnnotateCostsPassOptions costs;
        costs.model = "flops";
        costs.stablehloCost = options.prefer == "linalg" ? penalty : 0;
        costs.defaultCost = options.prefer == "stablehlo" ? penalty : 0;
        pm.addPass(createShloAnnotateCostsPass(costs));
        pm.addPass(equivalence::createEquivalenceSelectGreedy());
        equivalence::EquivalenceExtractOptions extract;
        extract.removeGraphs = true;
        pm.addPass(equivalence::createEquivalenceExtract(extract));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
}
