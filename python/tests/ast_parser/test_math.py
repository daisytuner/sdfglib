import docc
import pytest
import math


# =============================================================================
# Trigonometric Functions
# =============================================================================


def test_math_sin():
    @docc.program
    def math_sin(a) -> float:
        return math.sin(a)

    assert abs(math_sin(0.0) - 0.0) < 1e-6
    assert abs(math_sin(math.pi / 2) - 1.0) < 1e-6


def test_math_cos():
    @docc.program
    def math_cos(a) -> float:
        return math.cos(a)

    assert abs(math_cos(0.0) - 1.0) < 1e-6
    assert abs(math_cos(math.pi) - -1.0) < 1e-6


def test_math_tan():
    @docc.program
    def math_tan(a) -> float:
        return math.tan(a)

    assert abs(math_tan(0.0) - 0.0) < 1e-6
    assert abs(math_tan(math.pi / 4) - 1.0) < 1e-6


def test_math_asin():
    @docc.program
    def math_asin(a) -> float:
        return math.asin(a)

    assert abs(math_asin(0.0) - 0.0) < 1e-6
    assert abs(math_asin(1.0) - math.pi / 2) < 1e-6


def test_math_acos():
    @docc.program
    def math_acos(a) -> float:
        return math.acos(a)

    assert abs(math_acos(1.0) - 0.0) < 1e-6
    assert abs(math_acos(0.0) - math.pi / 2) < 1e-6


def test_math_atan():
    @docc.program
    def math_atan(a) -> float:
        return math.atan(a)

    assert abs(math_atan(0.0) - 0.0) < 1e-6
    assert abs(math_atan(1.0) - math.pi / 4) < 1e-6


def test_math_atan2():
    @docc.program
    def math_atan2(a, b) -> float:
        return math.atan2(a, b)

    assert abs(math_atan2(0.0, 1.0) - 0.0) < 1e-6
    assert abs(math_atan2(1.0, 1.0) - math.pi / 4) < 1e-6


# =============================================================================
# Hyperbolic Functions
# =============================================================================


def test_math_sinh():
    @docc.program
    def math_sinh(a) -> float:
        return math.sinh(a)

    assert abs(math_sinh(0.0) - 0.0) < 1e-6
    assert abs(math_sinh(1.0) - math.sinh(1.0)) < 1e-6


def test_math_cosh():
    @docc.program
    def math_cosh(a) -> float:
        return math.cosh(a)

    assert abs(math_cosh(0.0) - 1.0) < 1e-6
    assert abs(math_cosh(1.0) - math.cosh(1.0)) < 1e-6


def test_math_tanh():
    @docc.program
    def math_tanh(a) -> float:
        return math.tanh(a)

    assert abs(math_tanh(0.0) - 0.0) < 1e-6
    assert abs(math_tanh(1.0) - math.tanh(1.0)) < 1e-6


def test_math_asinh():
    @docc.program
    def math_asinh(a) -> float:
        return math.asinh(a)

    assert abs(math_asinh(0.0) - 0.0) < 1e-6
    assert abs(math_asinh(1.0) - math.asinh(1.0)) < 1e-6


def test_math_acosh():
    @docc.program
    def math_acosh(a) -> float:
        return math.acosh(a)

    assert abs(math_acosh(1.0) - 0.0) < 1e-6
    assert abs(math_acosh(2.0) - math.acosh(2.0)) < 1e-6


def test_math_atanh():
    @docc.program
    def math_atanh(a) -> float:
        return math.atanh(a)

    assert abs(math_atanh(0.0) - 0.0) < 1e-6
    assert abs(math_atanh(0.5) - math.atanh(0.5)) < 1e-6


# =============================================================================
# Exponential and Logarithmic Functions
# =============================================================================


def test_math_exp():
    @docc.program
    def math_exp(a) -> float:
        return math.exp(a)

    assert abs(math_exp(1.0) - math.e) < 1e-6
    assert abs(math_exp(0.0) - 1.0) < 1e-6


def test_math_exp2():
    @docc.program
    def math_exp2(a) -> float:
        return math.exp2(a)

    assert abs(math_exp2(0.0) - 1.0) < 1e-6
    assert abs(math_exp2(3.0) - 8.0) < 1e-6


def test_math_expm1():
    @docc.program
    def math_expm1(a) -> float:
        return math.expm1(a)

    assert abs(math_expm1(0.0) - 0.0) < 1e-6
    assert abs(math_expm1(1.0) - (math.e - 1)) < 1e-6


def test_math_log():
    @docc.program
    def math_log(a) -> float:
        return math.log(a)

    assert abs(math_log(math.e) - 1.0) < 1e-6
    assert abs(math_log(1.0) - 0.0) < 1e-6


def test_math_log2():
    @docc.program
    def math_log2(a) -> float:
        return math.log2(a)

    assert abs(math_log2(1.0) - 0.0) < 1e-6
    assert abs(math_log2(8.0) - 3.0) < 1e-6


def test_math_log10():
    @docc.program
    def math_log10(a) -> float:
        return math.log10(a)

    assert abs(math_log10(1.0) - 0.0) < 1e-6
    assert abs(math_log10(100.0) - 2.0) < 1e-6


def test_math_log1p():
    @docc.program
    def math_log1p(a) -> float:
        return math.log1p(a)

    assert abs(math_log1p(0.0) - 0.0) < 1e-6
    assert abs(math_log1p(math.e - 1) - 1.0) < 1e-6


# =============================================================================
# Power Functions
# =============================================================================


def test_math_pow():
    @docc.program
    def math_pow(a, b) -> float:
        return math.pow(a, b)

    assert abs(math_pow(2.0, 3.0) - 8.0) < 1e-6
    assert abs(math_pow(3.0, 2.0) - 9.0) < 1e-6


def test_math_sqrt():
    @docc.program
    def math_sqrt(a) -> float:
        return math.sqrt(a)

    assert abs(math_sqrt(4.0) - 2.0) < 1e-6
    assert abs(math_sqrt(9.0) - 3.0) < 1e-6


def test_math_cbrt():
    @docc.program
    def math_cbrt(a) -> float:
        return math.cbrt(a)

    assert abs(math_cbrt(8.0) - 2.0) < 1e-6
    assert abs(math_cbrt(27.0) - 3.0) < 1e-6


def test_math_hypot():
    @docc.program
    def math_hypot(a, b) -> float:
        return math.hypot(a, b)

    assert abs(math_hypot(3.0, 4.0) - 5.0) < 1e-6
    assert abs(math_hypot(5.0, 12.0) - 13.0) < 1e-6


# =============================================================================
# Rounding and Remainder Functions
# =============================================================================


def test_math_fabs():
    @docc.program
    def math_fabs(a) -> float:
        return math.fabs(a)

    assert abs(math_fabs(-5.0) - 5.0) < 1e-6
    assert abs(math_fabs(3.0) - 3.0) < 1e-6


def test_math_ceil():
    @docc.program
    def math_ceil(a) -> float:
        return math.ceil(a)

    assert abs(math_ceil(2.3) - 3.0) < 1e-6
    assert abs(math_ceil(-2.3) - -2.0) < 1e-6


def test_math_floor():
    @docc.program
    def math_floor(a) -> float:
        return math.floor(a)

    assert abs(math_floor(2.7) - 2.0) < 1e-6
    assert abs(math_floor(-2.7) - -3.0) < 1e-6


def test_math_trunc():
    @docc.program
    def math_trunc(a) -> float:
        return math.trunc(a)

    assert abs(math_trunc(2.7) - 2.0) < 1e-6
    assert abs(math_trunc(-2.7) - -2.0) < 1e-6


def test_math_fmod():
    @docc.program
    def math_fmod(a, b) -> float:
        return math.fmod(a, b)

    assert abs(math_fmod(5.0, 3.0) - 2.0) < 1e-6
    assert abs(math_fmod(-5.0, 3.0) - -2.0) < 1e-6


def test_math_remainder():
    @docc.program
    def math_remainder(a, b) -> float:
        return math.remainder(a, b)

    assert abs(math_remainder(5.0, 3.0) - -1.0) < 1e-6
    assert abs(math_remainder(7.0, 4.0) - -1.0) < 1e-6


# =============================================================================
# Floating-point Manipulation Functions
# =============================================================================


def test_math_copysign():
    @docc.program
    def math_copysign(a, b) -> float:
        return math.copysign(a, b)

    assert abs(math_copysign(5.0, -1.0) - -5.0) < 1e-6
    assert abs(math_copysign(-5.0, 1.0) - 5.0) < 1e-6


# =============================================================================
# Math Constants
# =============================================================================


def test_math_constants():
    @docc.program
    def math_pi() -> float:
        return math.pi

    assert abs(math_pi() - math.pi) < 1e-6

    @docc.program
    def math_e() -> float:
        return math.e

    assert abs(math_e() - math.e) < 1e-6
