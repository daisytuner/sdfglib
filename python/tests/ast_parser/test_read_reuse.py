import unittest
from unittest.mock import MagicMock
from docc.expression_visitor import ExpressionVisitor
from docc._sdfg import Scalar, PrimitiveType


class TestReadReuse(unittest.TestCase):
    def setUp(self):
        self.builder = MagicMock()
        # Mock exists to return True so _add_read takes the path calling add_access
        self.builder.exists.return_value = True

        self.visitor = ExpressionVisitor(builder=self.builder)

    def test_reuse_in_same_block(self):
        block = MagicMock()
        # Make block hashable for the cache dictionary
        block.__hash__.return_value = 1

        expr = "a"

        # First read
        self.visitor._add_read(block, expr)

        # Second read - should reuse
        self.visitor._add_read(block, expr)

        # Check add_access call count
        # It should be called exactly once with (block, "a", ...)
        # Note: add_access signature is (block, name, debug_info)

        # Filter calls to add_access
        access_calls = [
            call
            for call in self.builder.add_access.mock_calls
            if call.args[0] == block and call.args[1] == expr
        ]

        self.assertEqual(
            len(access_calls),
            1,
            "add_access should be called only once for repeated reads in same block",
        )

    def test_no_reuse_across_blocks(self):
        block1 = MagicMock()
        block1.__hash__.return_value = 1

        block2 = MagicMock()
        block2.__hash__.return_value = 2

        expr = "a"

        # Read in block1
        self.visitor._add_read(block1, expr)

        # Read in block2
        self.visitor._add_read(block2, expr)

        # Check add_access calls
        calls_block1 = [
            call
            for call in self.builder.add_access.mock_calls
            if call.args[0] == block1 and call.args[1] == expr
        ]
        calls_block2 = [
            call
            for call in self.builder.add_access.mock_calls
            if call.args[0] == block2 and call.args[1] == expr
        ]

        self.assertEqual(len(calls_block1), 1)
        self.assertEqual(len(calls_block2), 1)

    def test_reuse_with_subscript(self):
        # Test reuse for "a(i)" style strings which are handled differently in _add_read
        block = MagicMock()
        block.__hash__.return_value = 1

        expr = "a(i)"

        # First read
        self.visitor._add_read(block, expr)

        # Second read
        self.visitor._add_read(block, expr)

        # For "a(i)", it calls add_access with "a"
        name = "a"

        access_calls = [
            call
            for call in self.builder.add_access.mock_calls
            if call.args[0] == block and call.args[1] == name
        ]

        self.assertEqual(
            len(access_calls),
            1,
            "add_access should be called only once for repeated subscript reads",
        )

    def test_reuse_constant(self):
        block = MagicMock()
        block.__hash__.return_value = 1

        # Mock exists to return False so it treats it as constant
        self.builder.exists.return_value = False

        expr = "1.0"

        # First read
        self.visitor._add_read(block, expr)

        # Second read
        self.visitor._add_read(block, expr)

        # Check add_constant call count
        constant_calls = [
            call
            for call in self.builder.add_constant.mock_calls
            if call.args[0] == block and call.args[1] == expr
        ]

        self.assertEqual(
            len(constant_calls),
            1,
            "add_constant should be called only once for repeated constant reads",
        )


if __name__ == "__main__":
    unittest.main()
