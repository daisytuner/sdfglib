// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK-LABEL: test_constant
func.func @test_constant() -> i32 {
    %0 = sdfg.constant 1 : i32
    return %0 : i32
}