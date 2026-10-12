//===----------------------------------------------------------------------===//
// StableHLO-level rewrites.
//
//   python3 lower.py rewrites_base.pdl.mlir -o rewrites_base.mlir
//
// Matrix products are `stablehlo.dot_general` with contracting dims [1] x [0]
// and no batch dims. The type of a new intermediate product comes from the
// native DotResultType.
//===----------------------------------------------------------------------===//

// a + b -> b + a
pdl.pattern @add_comm : benefit(1) {
  %t = pdl.type
  %a = pdl.operand
  %b = pdl.operand
  %add = pdl.operation "stablehlo.add"(%a, %b : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  pdl.rewrite %add {
    %new = pdl.operation "stablehlo.add"(%b, %a : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    pdl.replace %add with %new
  }
}

// a*b + a*c -> a*(b + c)
pdl.pattern @factor_left : benefit(1) {
  %t = pdl.type
  %tb = pdl.type
  %a = pdl.operand
  %b = pdl.operand : %tb
  %c = pdl.operand : %tb
  %dn = pdl.attribute = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  %ab = pdl.operation "stablehlo.dot_general"(%a, %b : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  %abv = pdl.result 0 of %ab
  %ac = pdl.operation "stablehlo.dot_general"(%a, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  %acv = pdl.result 0 of %ac
  %add = pdl.operation "stablehlo.add"(%abv, %acv : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  pdl.rewrite %add {
    %bc = pdl.operation "stablehlo.add"(%b, %c : !pdl.value, !pdl.value) -> (%tb : !pdl.type)
    %bcv = pdl.result 0 of %bc
    %new = pdl.operation "stablehlo.dot_general"(%a, %bcv : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
    pdl.replace %add with %new
  }
}

// a*c + b*c -> (a + b)*c
pdl.pattern @factor_right : benefit(1) {
  %t = pdl.type
  %ta = pdl.type
  %a = pdl.operand : %ta
  %b = pdl.operand : %ta
  %c = pdl.operand
  %dn = pdl.attribute = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  %ac = pdl.operation "stablehlo.dot_general"(%a, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  %acv = pdl.result 0 of %ac
  %bc = pdl.operation "stablehlo.dot_general"(%b, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  %bcv = pdl.result 0 of %bc
  %add = pdl.operation "stablehlo.add"(%acv, %bcv : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  pdl.rewrite %add {
    %ab = pdl.operation "stablehlo.add"(%a, %b : !pdl.value, !pdl.value) -> (%ta : !pdl.type)
    %abv = pdl.result 0 of %ab
    %new = pdl.operation "stablehlo.dot_general"(%abv, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
    pdl.replace %add with %new
  }
}

// transpose(transpose(a)) -> a
pdl.pattern @transpose_involution : benefit(1) {
  %t = pdl.type
  %ti = pdl.type
  %a = pdl.operand : %t
  %perm = pdl.attribute = array<i64: 1, 0>
  %inner = pdl.operation "stablehlo.transpose"(%a : !pdl.value) {"permutation" = %perm} -> (%ti : !pdl.type)
  %innerv = pdl.result 0 of %inner
  %outer = pdl.operation "stablehlo.transpose"(%innerv : !pdl.value) {"permutation" = %perm} -> (%t : !pdl.type)
  pdl.rewrite %outer {
    pdl.replace %outer with (%a : !pdl.value)
  }
}

// (a*b)*c -> a*(b*c)
pdl.pattern @dot_assoc_l2r : benefit(1) {
  %t = pdl.type
  %tab = pdl.type
  %a = pdl.operand
  %b = pdl.operand
  %c = pdl.operand
  %dn = pdl.attribute = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  %ab = pdl.operation "stablehlo.dot_general"(%a, %b : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%tab : !pdl.type)
  %abv = pdl.result 0 of %ab
  %abc = pdl.operation "stablehlo.dot_general"(%abv, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  pdl.rewrite %abc {
    %tbc = pdl.apply_native_rewrite "DotResultType"(%b, %c : !pdl.value, !pdl.value) : !pdl.type
    %bc = pdl.operation "stablehlo.dot_general"(%b, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%tbc : !pdl.type)
    %bcv = pdl.result 0 of %bc
    %new = pdl.operation "stablehlo.dot_general"(%a, %bcv : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
    pdl.replace %abc with %new
  }
}

// a*(b*c) -> (a*b)*c
pdl.pattern @dot_assoc_r2l : benefit(1) {
  %t = pdl.type
  %tbc = pdl.type
  %a = pdl.operand
  %b = pdl.operand
  %c = pdl.operand
  %dn = pdl.attribute = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  %bc = pdl.operation "stablehlo.dot_general"(%b, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%tbc : !pdl.type)
  %bcv = pdl.result 0 of %bc
  %abc = pdl.operation "stablehlo.dot_general"(%a, %bcv : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  pdl.rewrite %abc {
    %tab = pdl.apply_native_rewrite "DotResultType"(%a, %b : !pdl.value, !pdl.value) : !pdl.type
    %ab = pdl.operation "stablehlo.dot_general"(%a, %b : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%tab : !pdl.type)
    %abv = pdl.result 0 of %ab
    %new = pdl.operation "stablehlo.dot_general"(%abv, %c : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
    pdl.replace %abc with %new
  }
}

// transpose(a*b) -> transpose(b)*transpose(a)
pdl.pattern @transpose_of_dot : benefit(1) {
  %t = pdl.type
  %a = pdl.operand : %t
  %b = pdl.operand : %t
  %dn = pdl.attribute = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  %perm = pdl.attribute = array<i64: 1, 0>
  %ab = pdl.operation "stablehlo.dot_general"(%a, %b : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
  %abv = pdl.result 0 of %ab
  %tr = pdl.operation "stablehlo.transpose"(%abv : !pdl.value) {"permutation" = %perm} -> (%t : !pdl.type)
  pdl.rewrite %tr {
    %bt = pdl.operation "stablehlo.transpose"(%b : !pdl.value) {"permutation" = %perm} -> (%t : !pdl.type)
    %btv = pdl.result 0 of %bt
    %at = pdl.operation "stablehlo.transpose"(%a : !pdl.value) {"permutation" = %perm} -> (%t : !pdl.type)
    %atv = pdl.result 0 of %at
    %new = pdl.operation "stablehlo.dot_general"(%btv, %atv : !pdl.value, !pdl.value) {"dot_dimension_numbers" = %dn} -> (%t : !pdl.type)
    pdl.replace %tr with %new
  }
}
