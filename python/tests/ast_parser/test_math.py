import docc
import pytest
import math


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


def test_math_sqrt():
    @docc.program
    def math_sqrt(a) -> float:
        return math.sqrt(a)

    assert abs(math_sqrt(4.0) - 2.0) < 1e-6
    assert abs(math_sqrt(9.0) - 3.0) < 1e-6


def test_math_exp():
    @docc.program
    def math_exp(a) -> float:
        return math.exp(a)

    assert abs(math_exp(1.0) - math.e) < 1e-6
    assert abs(math_exp(0.0) - 1.0) < 1e-6


def test_math_log():
    @docc.program
    def math_log(a) -> float:
        return math.log(a)

    assert abs(math_log(math.e) - 1.0) < 1e-6
    assert abs(math_log(1.0) - 0.0) < 1e-6


def test_math_pow():
    @docc.program
    def math_pow(a, b) -> float:
        return math.pow(a, b)

    assert abs(math_pow(2.0, 3.0) - 8.0) < 1e-6
    assert abs(math_pow(3.0, 2.0) - 9.0) < 1e-6


def test_math_constants():
    @docc.program
    def math_pi() -> float:
        return math.pi

    assert abs(math_pi() - math.pi) < 1e-6

    @docc.program
    def math_e() -> float:
        return math.e

    assert abs(math_e() - math.e) < 1e-6
