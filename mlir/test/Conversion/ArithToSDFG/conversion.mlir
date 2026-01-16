// RUN: sdfg-opt %s --convert-arith-to-sdfg > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_constant
func.func @test_constant() -> i32 {
    // CHECK-NEXT: %[[CONST:.*]] = sdfg.constant 1 : i32
    %0 = arith.constant 1 : i32
    // CHECK-NEXT: return %[[CONST]] : i32
    return %0 : i32
}

// CHECK-LABEL: @test_constant_type_failure
func.func @test_constant_type_failure() -> i4 {
    // CHECK-NEXT: %[[CONST:.*]] = arith.constant 1 : i4
    %0 = arith.constant 1 : i4
    // CHECK-NEXT: return %[[CONST]] : i4
    return %0 : i4
}