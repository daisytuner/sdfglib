import ast
from ._sdfg import Scalar, PrimitiveType, Pointer
from .ast_utils import get_debug_info


class ONNXHandler:
    def __init__(self, builder, array_info, symbol_table, expr_visitor):
        self.builder = builder
        self.array_info = array_info
        self.symbol_table = symbol_table
        self.expr_visitor = expr_visitor

    def _parse_expr(self, node):
        return self.expr_visitor.visit(node)

    def _parse_perm(self, node):
        # Parse list or tuple of integers
        if isinstance(node, (ast.List, ast.Tuple)):
            # Warning: this relies on visit returning string '1', '2' etc. and casting to int
            # Proper way is checking for Constant(value=int)
            res = []
            for elt in node.elts:
                val = self.expr_visitor.visit(elt)
                res.append(int(val))
            return res
        return []

    # Transpose Support
    def is_transpose(self, node):
        # Case 1: np.transpose(arr, ...)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "transpose":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "transpose":
                return True

        # Case 2: arr.T
        if isinstance(node, ast.Attribute) and node.attr == "T":
            return True

        return False

    def handle_transpose(self, target, value_node):
        if not self.is_transpose(value_node):
            return False

        # Extract input
        input_node = None
        perm = []

        if isinstance(value_node, ast.Attribute) and value_node.attr == "T":
            input_node = value_node.value
            # For .T, permutation is reverse
            perm = []  # Empty means reverse logic to be applied later

        elif isinstance(value_node, ast.Call):
            args = value_node.args
            keywords = value_node.keywords

            # Identify if function or method
            is_numpy_func = False
            if isinstance(value_node.func, ast.Attribute):
                # Heuristic: if called on 'np' or 'numpy'
                caller = ""
                if isinstance(value_node.func.value, ast.Name):
                    caller = value_node.func.value.id

                if caller in ["np", "numpy"]:
                    is_numpy_func = True

            elif isinstance(value_node.func, ast.Name):
                # Assumed imported as function
                is_numpy_func = True

            if is_numpy_func:
                # np.transpose(arr, axes=...)
                if len(args) < 1:
                    return False
                input_node = args[0]
                if len(args) > 1:
                    perm = self._parse_perm(args[1])

                for kw in keywords:
                    if kw.arg == "axes":
                        perm = self._parse_perm(kw.value)

            else:
                # Method call: arr.transpose(axes=...)
                if isinstance(value_node.func, ast.Attribute):
                    input_node = value_node.func.value
                else:
                    return False  # processing error

                if len(args) > 0:
                    perm = self._parse_perm(args[0])
                for kw in keywords:
                    if kw.arg == "axes":
                        perm = self._parse_perm(kw.value)

        # Process Input
        input_name = self._parse_expr(input_node)
        if input_name not in self.array_info:
            # Fallback if input is not a tracked array?
            return False

        in_info = self.array_info[input_name]
        in_shape = in_info["shapes"]
        in_strings = [str(s) for s in in_shape]

        # Calculate Permutation if empty
        if not perm:
            perm = list(range(len(in_shape)))[::-1]

        out_shape = [in_strings[p] for p in perm]

        # Register Output
        target_name = ""
        if isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, str):
            target_name = target

        dtype = Scalar(PrimitiveType.Double)  # Default
        # Infer dtype from input
        if input_name in self.symbol_table:
            input_type = self.symbol_table[input_name]
            if isinstance(input_type, Pointer):
                dtype = input_type.pointee_type
            else:
                dtype = input_type
            # Arrays are usually Pointers to Scalars.

        ptr_type = Pointer(dtype)

        if not self.builder.exists(target_name):
            self.builder.add_container(target_name, ptr_type, False)
            self.symbol_table[target_name] = ptr_type
            self.array_info[target_name] = {"ndim": len(out_shape), "shapes": out_shape}

            # Allocate memory
            block_alloc = self.builder.add_block()
            size_expr = "1"
            for dim in out_shape:
                size_expr = f"({size_expr} * {dim})"
            element_size = self.builder.get_sizeof(dtype)
            total_size = f"({size_expr} * {element_size})"

            t_malloc = self.builder.add_malloc(block_alloc, total_size)
            t_ptr = self.builder.add_access(block_alloc, target_name)
            self.builder.add_memlet(
                block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type
            )

        debug_info = get_debug_info(
            value_node, getattr(self.builder, "filename", ""), ""
        )

        self.builder.add_transpose(
            input_name, target_name, in_strings, perm, debug_info
        )
        return True
