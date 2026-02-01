import ast
import pytest
from docc.python import get_debug_info


def test_get_debug_info():
    code = "a = 1"
    tree = ast.parse(code)
    node = tree.body[0]  # Assign

    filename = "test.py"
    func_name = "test_func"

    di = get_debug_info(node, filename, func_name)

    assert di.filename == filename
    assert di.function == func_name
    assert di.start_line == 1
    # col_offset is 0 for 'a', so start_column should be 1
    assert di.start_column == 1
    # end_lineno should be 1
    assert di.end_line == 1
    # end_col_offset should be 5 (length of "a = 1")
    assert di.end_column == 6
