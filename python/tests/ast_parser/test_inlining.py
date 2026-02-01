from docc.python import native
import pytest


def helper_add(a, b):
    return a + b


def test_simple_inlining():
    @native
    def simple_inlining(a, b):
        return helper_add(a, b)

    assert simple_inlining(10, 20) == 30


def helper_nested(x):
    return x * 2


def helper_outer(x):
    return helper_nested(x) + 1


def test_nested_inlining():
    @native
    def nested_inlining(x):
        return helper_outer(x)

    assert nested_inlining(10) == 21


def helper_with_local_var(x):
    y = 10
    return x + y


def test_inlining_local_vars():
    @native
    def inlining_local_vars(x):
        y = 5
        return helper_with_local_var(x) + y

    assert inlining_local_vars(20) == 35  # (20 + 10) + 5
