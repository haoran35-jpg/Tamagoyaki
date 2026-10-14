// RUN: shlo-mlir-opt %s -equivalence-insert-graph="nested=false" \
// RUN:   --shlo-saturate="patterns-file=%rules/rewrites_multi.mlir max-iters=4" \
// RUN:   --canonicalize --cse --equivalence-restore-invariants \
// RUN:   --shlo-extract="prefer=linalg" --mlir-print-local-scope \
// RUN: | FileCheck %s --implicit-check-not=stablehlo. \
// RUN:     --implicit-check-not=equivalence. --implicit-check-not=linalg.transpose

// RUN: shlo-mlir-opt %s -equivalence-insert-graph="nested=false" \
// RUN:   --shlo-saturate="patterns-file=%rules/rewrites_multi.mlir max-iters=4" \
// RUN:   --equivalence-graph-size -o /dev/null > %t.size4
// RUN: shlo-mlir-opt %s -equivalence-insert-graph="nested=false" \
// RUN:   --shlo-saturate="patterns-file=%rules/rewrites_multi.mlir max-iters=8" \
// RUN:   --equivalence-graph-size -o /dev/null > %t.size8
// RUN: diff %t.size4 %t.size8

// Linalg-level rules (rewrites_linalg.pdl.mlir) run in the same saturation as
// the StableHLO rules and the lowering. A transpose feeding a matmul is folded
// into the matmul's indexing map; with the transpose copy gone, the extraction
// prefers the folded matmul.

// CHECK-LABEL: func.func @transpose_lhs(
// CHECK-SAME:      %[[A:[a-z0-9]+]]: tensor<8x4xf32>, %[[B:[a-z0-9]+]]: tensor<8x16xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%[[A]], %[[B]] :
// CHECK-NOT:     linalg.generic
// CHECK:         return
func.func @transpose_lhs(%a: tensor<8x4xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %at = stablehlo.transpose %a, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %d = stablehlo.dot_general %at, %b, contracting_dims = [1] x [0]
      : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %d : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @transpose_rhs(
// CHECK-SAME:      %[[A:[a-z0-9]+]]: tensor<4x8xf32>, %[[B:[a-z0-9]+]]: tensor<16x8xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%[[A]], %[[B]] :
// CHECK-NOT:     linalg.generic
// CHECK:         return
func.func @transpose_rhs(%a: tensor<4x8xf32>, %b: tensor<16x8xf32>) -> tensor<4x16xf32> {
  %bt = stablehlo.transpose %b, dims = [1, 0] : (tensor<16x8xf32>) -> tensor<8x16xf32>
  %d = stablehlo.dot_general %a, %bt, contracting_dims = [1] x [0]
      : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %d : tensor<4x16xf32>
}

// A matmul that already reads a transposed operand is not folded again.
// CHECK-LABEL: func.func @both
// CHECK-COUNT-1: linalg.matmul
// CHECK-NOT:     linalg.matmul
// CHECK:         return
func.func @both(%a: tensor<8x4xf32>, %b: tensor<16x8xf32>) -> tensor<4x16xf32> {
  %at = stablehlo.transpose %a, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %bt = stablehlo.transpose %b, dims = [1, 0] : (tensor<16x8xf32>) -> tensor<8x16xf32>
  %d = stablehlo.dot_general %at, %bt, contracting_dims = [1] x [0]
      : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %d : tensor<4x16xf32>
}
