import ast
import inspect
import textwrap
from ._sdfg import (
    Scalar,
    PrimitiveType,
    Pointer,
    Type,
    DebugInfo,
    Structure,
    TaskletCode,
    CMathFunction,
)


class ExpressionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        array_info=None,
        builder=None,
        symbol_table=None,
        globals_dict=None,
        inliner=None,
        unique_counter_ref=None,
        structure_member_info=None,
    ):
        self.array_info = array_info if array_info is not None else {}
        self.builder = builder
        self.symbol_table = symbol_table if symbol_table is not None else {}
        self.globals_dict = globals_dict if globals_dict is not None else {}
        self.inliner = inliner
        self._unique_counter_ref = (
            unique_counter_ref if unique_counter_ref is not None else [0]
        )
        self._access_cache = {}
        self.la_handler = None
        self.structure_member_info = (
            structure_member_info if structure_member_info is not None else {}
        )
        self._init_numpy_handlers()

    def _get_unique_id(self):
        self._unique_counter_ref[0] += 1
        return self._unique_counter_ref[0]

    def _get_temp_name(self, prefix="_tmp_"):
        if hasattr(self.builder, "find_new_name"):
            return self.builder.find_new_name(prefix)
        return f"{prefix}{self._get_unique_id()}"

    def _is_indirect_access(self, node):
        """Check if a node represents an indirect array access (e.g., A[B[i]]).

        Returns True if the node is a subscript where the index itself is a subscript
        into an array (indirect access pattern).
        """
        if not isinstance(node, ast.Subscript):
            return False
        # Check if value is a subscripted array access
        if isinstance(node.value, ast.Name):
            arr_name = node.value.id
            if arr_name in self.array_info:
                # Check if slice/index is itself an array access
                if isinstance(node.slice, ast.Subscript):
                    if isinstance(node.slice.value, ast.Name):
                        idx_arr_name = node.slice.value.id
                        if idx_arr_name in self.array_info:
                            return True
        return False

    def _contains_indirect_access(self, node):
        """Check if an AST node contains any indirect array access.

        Used to detect expressions like A_row[i] that would be used as slice bounds.
        """
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                arr_name = node.value.id
                if arr_name in self.array_info:
                    return True
        elif isinstance(node, ast.BinOp):
            return self._contains_indirect_access(
                node.left
            ) or self._contains_indirect_access(node.right)
        elif isinstance(node, ast.UnaryOp):
            return self._contains_indirect_access(node.operand)
        return False

    def _materialize_indirect_access(
        self, node, debug_info=None, return_original_expr=False
    ):
        """Materialize an array access into a scalar variable using tasklet+memlets.

        For indirect memory access patterns in SDFGs, we need to:
        1. Create a scalar container for the result
        2. Create a tasklet that performs the assignment
        3. Use memlets to read from the array and write to the scalar
        4. Return the scalar name (which can be used as a symbolic expression)

        This is the canonical SDFG pattern for indirect access.

        If return_original_expr is True, also returns the original array access
        expression using parentheses notation (e.g., "A_row(0)") which is consistent
        with SDFG subset notation. The runtime evaluator will convert this to
        bracket notation for Python evaluation.
        """
        if not self.builder:
            # Without builder, just return the expression string
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        if debug_info is None:
            debug_info = DebugInfo()

        if not isinstance(node, ast.Subscript):
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        if not isinstance(node.value, ast.Name):
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        arr_name = node.value.id
        if arr_name not in self.array_info:
            expr = self.visit(node)
            return (expr, expr) if return_original_expr else expr

        # Determine the element type
        dtype = Scalar(PrimitiveType.Int64)  # Default for indices
        if arr_name in self.symbol_table:
            t = self.symbol_table[arr_name]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        # Create scalar container for the result
        tmp_name = self._get_temp_name("_idx_")
        self.builder.add_container(tmp_name, dtype, False)
        self.symbol_table[tmp_name] = dtype

        # Get the index expression
        ndim = self.array_info[arr_name]["ndim"]
        shapes = self.array_info[arr_name].get("shapes", [])

        # Compute linear index from the subscript
        if isinstance(node.slice, ast.Tuple):
            indices = [self.visit(elt) for elt in node.slice.elts]
        else:
            indices = [self.visit(node.slice)]

        # Handle cases where we need recursive materialization
        materialized_indices = []
        for i, idx_str in enumerate(indices):
            # Check if the index itself needs materialization (nested indirect)
            # This happens when idx_str looks like an array access e.g., "arr(i)"
            if "(" in idx_str and idx_str.endswith(")"):
                # This is an array access, it should already be a valid symbolic expression
                # or a scalar variable name
                materialized_indices.append(idx_str)
            else:
                materialized_indices.append(idx_str)

        # Compute linear index
        linear_index = self._compute_linear_index(
            materialized_indices, shapes, arr_name, ndim
        )

        # Create block with tasklet and memlets
        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, arr_name, debug_info)
        t_dst = self.builder.add_access(block, tmp_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", linear_index, None, debug_info
        )
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", "", None, debug_info
        )

        if return_original_expr:
            # Return both the materialized variable name and the original array access expression
            # Use parentheses notation which is consistent with SDFG subset syntax
            original_expr = f"{arr_name}({linear_index})"
            return (tmp_name, original_expr)

        return tmp_name

    def _init_numpy_handlers(self):
        self.numpy_handlers = {
            "empty": self._handle_numpy_alloc,
            "empty_like": self._handle_numpy_empty_like,
            "zeros": self._handle_numpy_alloc,
            "zeros_like": self._handle_numpy_zeros_like,
            "ones": self._handle_numpy_alloc,
            "ndarray": self._handle_numpy_alloc,  # np.ndarray() constructor
            "eye": self._handle_numpy_eye,
            "add": self._handle_numpy_binary_op,
            "subtract": self._handle_numpy_binary_op,
            "multiply": self._handle_numpy_binary_op,
            "divide": self._handle_numpy_binary_op,
            "power": self._handle_numpy_binary_op,
            "exp": self._handle_numpy_unary_op,
            "abs": self._handle_numpy_unary_op,
            "absolute": self._handle_numpy_unary_op,
            "sqrt": self._handle_numpy_unary_op,
            "tanh": self._handle_numpy_unary_op,
            "sum": self._handle_numpy_reduce,
            "max": self._handle_numpy_reduce,
            "min": self._handle_numpy_reduce,
            "mean": self._handle_numpy_reduce,
            "std": self._handle_numpy_reduce,
            "matmul": self._handle_numpy_matmul,
            "dot": self._handle_numpy_matmul,
            "matvec": self._handle_numpy_matmul,
            "outer": self._handle_numpy_outer,
            "minimum": self._handle_numpy_binary_op,
            "maximum": self._handle_numpy_binary_op,
            "where": self._handle_numpy_where,
        }

    def generic_visit(self, node):
        return super().generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def visit_Name(self, node):
        name = node.id
        # Check if it's a global constant (not a local variable/array)
        if name not in self.symbol_table and self.globals_dict is not None:
            if name in self.globals_dict:
                val = self.globals_dict[name]
                # Only substitute simple numeric constants
                if isinstance(val, (int, float)):
                    return str(val)
        return name

    def _map_numpy_dtype(self, dtype_node):
        # Default to double
        if dtype_node is None:
            return Scalar(PrimitiveType.Double)

        if isinstance(dtype_node, ast.Name):
            if dtype_node.id == "float":
                return Scalar(PrimitiveType.Double)
            if dtype_node.id == "int":
                return Scalar(PrimitiveType.Int64)
            if dtype_node.id == "bool":
                return Scalar(PrimitiveType.Bool)

        if isinstance(dtype_node, ast.Attribute):
            # Handle array.dtype
            if (
                isinstance(dtype_node.value, ast.Name)
                and dtype_node.value.id in self.symbol_table
                and dtype_node.attr == "dtype"
            ):
                sym_type = self.symbol_table[dtype_node.value.id]
                if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                    return sym_type.pointee_type

            if isinstance(dtype_node.value, ast.Name) and dtype_node.value.id in [
                "numpy",
                "np",
            ]:
                if dtype_node.attr == "float64":
                    return Scalar(PrimitiveType.Double)
                if dtype_node.attr == "float32":
                    return Scalar(PrimitiveType.Float)
                if dtype_node.attr == "int64":
                    return Scalar(PrimitiveType.Int64)
                if dtype_node.attr == "int32":
                    return Scalar(PrimitiveType.Int32)
                if dtype_node.attr == "bool_":
                    return Scalar(PrimitiveType.Bool)

        # Fallback
        return Scalar(PrimitiveType.Double)

    def _is_int(self, operand):
        try:
            if operand.lstrip("-").isdigit():
                return True
        except ValueError:
            pass

        name = operand
        if "(" in operand and operand.endswith(")"):
            name = operand.split("(")[0]

        if name in self.symbol_table:
            t = self.symbol_table[name]

            def is_int_ptype(pt):
                return pt in [
                    PrimitiveType.Int64,
                    PrimitiveType.Int32,
                    PrimitiveType.Int8,
                    PrimitiveType.Int16,
                    PrimitiveType.UInt64,
                    PrimitiveType.UInt32,
                    PrimitiveType.UInt8,
                    PrimitiveType.UInt16,
                ]

            if isinstance(t, Scalar):
                return is_int_ptype(t.primitive_type)

            if type(t).__name__ == "Array" and hasattr(t, "element_type"):
                et = t.element_type
                if callable(et):
                    et = et()
                if isinstance(et, Scalar):
                    return is_int_ptype(et.primitive_type)

            if type(t).__name__ == "Pointer":
                if hasattr(t, "pointee_type"):
                    et = t.pointee_type
                    if callable(et):
                        et = et()
                    if isinstance(et, Scalar):
                        return is_int_ptype(et.primitive_type)
                # Fallback: check if it has element_type (maybe alias?)
                if hasattr(t, "element_type"):
                    et = t.element_type
                    if callable(et):
                        et = et()
                    if isinstance(et, Scalar):
                        return is_int_ptype(et.primitive_type)

        return False

    def _add_read(self, block, expr_str, debug_info=None):
        # Try to reuse access node
        try:
            if (block, expr_str) in self._access_cache:
                return self._access_cache[(block, expr_str)]
        except TypeError:
            # block might not be hashable
            pass

        if debug_info is None:
            debug_info = DebugInfo()

        if "(" in expr_str and expr_str.endswith(")"):
            name = expr_str.split("(")[0]
            subset = expr_str[expr_str.find("(") + 1 : -1]
            access = self.builder.add_access(block, name, debug_info)
            try:
                self._access_cache[(block, expr_str)] = (access, subset)
            except TypeError:
                pass
            return access, subset

        if self.builder.exists(expr_str):
            access = self.builder.add_access(block, expr_str, debug_info)
            # For pointer types representing 0-D arrays, dereference with "0"
            subset = ""
            if expr_str in self.symbol_table:
                sym_type = self.symbol_table[expr_str]
                if isinstance(sym_type, Pointer):
                    # Check if it's a 0-D array (scalar wrapped in pointer)
                    if expr_str in self.array_info:
                        ndim = self.array_info[expr_str].get("ndim", 0)
                        if ndim == 0:
                            subset = "0"
                    else:
                        # Pointer without array_info is treated as 0-D
                        subset = "0"
            try:
                self._access_cache[(block, expr_str)] = (access, subset)
            except TypeError:
                pass
            return access, subset

        dtype = Scalar(PrimitiveType.Double)
        if self._is_int(expr_str):
            dtype = Scalar(PrimitiveType.Int64)
        elif expr_str == "true" or expr_str == "false":
            dtype = Scalar(PrimitiveType.Bool)

        const_node = self.builder.add_constant(block, expr_str, dtype, debug_info)
        try:
            self._access_cache[(block, expr_str)] = (const_node, "")
        except TypeError:
            pass
        return const_node, ""

    def _handle_min_max(self, node, func_name):
        args = [self.visit(arg) for arg in node.args]
        if len(args) != 2:
            raise NotImplementedError(f"{func_name} only supported with 2 arguments")

        # Check types
        is_float = False
        arg_types = []

        for arg in args:
            name = arg
            if "(" in arg and arg.endswith(")"):
                name = arg.split("(")[0]

            if name in self.symbol_table:
                t = self.symbol_table[name]
                if isinstance(t, Pointer):
                    t = t.base_type

                if t.primitive_type == PrimitiveType.Double:
                    is_float = True
                    arg_types.append(PrimitiveType.Double)
                else:
                    arg_types.append(PrimitiveType.Int64)
            elif self._is_int(arg):
                arg_types.append(PrimitiveType.Int64)
            else:
                # Assume float constant
                is_float = True
                arg_types.append(PrimitiveType.Double)

        dtype = Scalar(PrimitiveType.Double if is_float else PrimitiveType.Int64)

        tmp_name = self._get_temp_name("_tmp_")
        self.builder.add_container(tmp_name, dtype, False)
        self.symbol_table[tmp_name] = dtype

        if is_float:
            # Cast args if necessary
            casted_args = []
            for i, arg in enumerate(args):
                if arg_types[i] != PrimitiveType.Double:
                    # Create temp double
                    tmp_cast = self._get_temp_name("_cast_")
                    self.builder.add_container(
                        tmp_cast, Scalar(PrimitiveType.Double), False
                    )
                    self.symbol_table[tmp_cast] = Scalar(PrimitiveType.Double)

                    # Assign int to double (implicit cast)
                    self.builder.add_assignment(tmp_cast, arg)
                    casted_args.append(tmp_cast)
                else:
                    casted_args.append(arg)

            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            intrinsic_name = (
                CMathFunction.fmax if func_name == "max" else CMathFunction.fmin
            )
            t_task = self.builder.add_cmath(block, intrinsic_name)

            for i, arg in enumerate(casted_args):
                t_arg, arg_sub = self._add_read(block, arg)
                self.builder.add_memlet(
                    block, t_arg, "void", t_task, f"_in{i+1}", arg_sub
                )
        else:
            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            # Use int_smax/int_smin tasklet
            opcode = None
            if func_name == "max":
                opcode = TaskletCode.int_smax
            else:
                opcode = TaskletCode.int_smin
            t_task = self.builder.add_tasklet(block, opcode, ["_in1", "_in2"], ["_out"])

            for i, arg in enumerate(args):
                t_arg, arg_sub = self._add_read(block, arg)
                self.builder.add_memlet(
                    block, t_arg, "void", t_task, f"_in{i+1}", arg_sub
                )

        self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")
        return tmp_name

    def _handle_python_cast(self, node, func_name):
        """Handle Python type casts: int(), float(), bool()"""
        if len(node.args) != 1:
            raise NotImplementedError(f"{func_name}() cast requires exactly 1 argument")

        arg = self.visit(node.args[0])

        # Determine target type based on cast function
        if func_name == "int":
            target_dtype = Scalar(PrimitiveType.Int64)
        elif func_name == "float":
            target_dtype = Scalar(PrimitiveType.Double)
        elif func_name == "bool":
            target_dtype = Scalar(PrimitiveType.Bool)
        else:
            raise NotImplementedError(f"Cast to {func_name} not supported")

        # Determine source type
        source_dtype = None
        name = arg
        if "(" in arg and arg.endswith(")"):
            name = arg.split("(")[0]

        if name in self.symbol_table:
            source_dtype = self.symbol_table[name]
            if isinstance(source_dtype, Pointer):
                source_dtype = source_dtype.base_type
        elif self._is_int(arg):
            source_dtype = Scalar(PrimitiveType.Int64)
        elif arg == "true" or arg == "false":
            source_dtype = Scalar(PrimitiveType.Bool)
        else:
            # Assume float constant
            source_dtype = Scalar(PrimitiveType.Double)

        # Create temporary variable for result
        tmp_name = self._get_temp_name("_tmp_")
        self.builder.add_container(tmp_name, target_dtype, False)
        self.symbol_table[tmp_name] = target_dtype

        # Use tasklet assign opcode for casting (as specified in problem statement)
        block = self.builder.add_block()
        t_src, src_sub = self._add_read(block, arg)
        t_dst = self.builder.add_access(block, tmp_name)
        t_task = self.builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
        self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
        self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

        return tmp_name

    def visit_Call(self, node):
        func_name = ""
        module_name = ""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "math":
                    module_name = "math"
                    func_name = node.func.attr
                elif node.func.value.id in ["numpy", "np"]:
                    module_name = "numpy"
                    func_name = node.func.attr
                else:
                    # Check if it's a method call on an array (e.g., arr.astype(...), arr.copy())
                    array_name = node.func.value.id
                    method_name = node.func.attr
                    if array_name in self.array_info and method_name == "astype":
                        return self._handle_numpy_astype(node, array_name)
                    elif array_name in self.array_info and method_name == "copy":
                        return self._handle_numpy_copy(node, array_name)
            elif isinstance(node.func.value, ast.Attribute):
                if (
                    isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "scipy"
                    and node.func.value.attr == "special"
                ):
                    if node.func.attr == "softmax":
                        return self._handle_scipy_softmax(node, "softmax")
                # Handle np.add.outer, np.subtract.outer, np.multiply.outer, etc.
                elif (
                    isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id in ["numpy", "np"]
                    and node.func.attr == "outer"
                ):
                    ufunc_name = node.func.value.attr  # "add", "subtract", etc.
                    return self._handle_ufunc_outer(node, ufunc_name)

        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if module_name == "numpy":
            if func_name in self.numpy_handlers:
                return self.numpy_handlers[func_name](node, func_name)

        if func_name in ["max", "min"]:
            return self._handle_min_max(node, func_name)

        # Handle Python type casts (int, float, bool)
        if func_name in ["int", "float", "bool"]:
            return self._handle_python_cast(node, func_name)

        math_funcs = {
            # Trigonometric functions
            "sin": CMathFunction.sin,
            "cos": CMathFunction.cos,
            "tan": CMathFunction.tan,
            "asin": CMathFunction.asin,
            "acos": CMathFunction.acos,
            "atan": CMathFunction.atan,
            "atan2": CMathFunction.atan2,
            # Hyperbolic functions
            "sinh": CMathFunction.sinh,
            "cosh": CMathFunction.cosh,
            "tanh": CMathFunction.tanh,
            "asinh": CMathFunction.asinh,
            "acosh": CMathFunction.acosh,
            "atanh": CMathFunction.atanh,
            # Exponential and logarithmic functions
            "exp": CMathFunction.exp,
            "exp2": CMathFunction.exp2,
            "expm1": CMathFunction.expm1,
            "log": CMathFunction.log,
            "log2": CMathFunction.log2,
            "log10": CMathFunction.log10,
            "log1p": CMathFunction.log1p,
            # Power functions
            "pow": CMathFunction.pow,
            "sqrt": CMathFunction.sqrt,
            "cbrt": CMathFunction.cbrt,
            "hypot": CMathFunction.hypot,
            # Rounding and remainder functions
            "abs": CMathFunction.fabs,
            "fabs": CMathFunction.fabs,
            "ceil": CMathFunction.ceil,
            "floor": CMathFunction.floor,
            "trunc": CMathFunction.trunc,
            "fmod": CMathFunction.fmod,
            "remainder": CMathFunction.remainder,
            # Floating-point manipulation functions
            "copysign": CMathFunction.copysign,
            # Other functions
            "fma": CMathFunction.fma,
        }

        if func_name in math_funcs:
            args = [self.visit(arg) for arg in node.args]

            tmp_name = self._get_temp_name("_tmp_")
            dtype = Scalar(PrimitiveType.Double)
            self.builder.add_container(tmp_name, dtype, False)
            self.symbol_table[tmp_name] = dtype

            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            t_task = self.builder.add_cmath(block, math_funcs[func_name])

            for i, arg in enumerate(args):
                t_arg, arg_sub = self._add_read(block, arg)
                self.builder.add_memlet(
                    block, t_arg, "void", t_task, f"_in{i+1}", arg_sub
                )

            self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")
            return tmp_name

        if func_name in self.globals_dict:
            obj = self.globals_dict[func_name]
            if inspect.isfunction(obj):
                return self._handle_inline_call(node, obj)

        raise NotImplementedError(f"Function call {func_name} not supported")

    def _handle_inline_call(self, node, func_obj):
        # 1. Parse function source
        try:
            source_lines, start_line = inspect.getsourcelines(func_obj)
            source = textwrap.dedent("".join(source_lines))
            tree = ast.parse(source)
            func_def = tree.body[0]
        except Exception as e:
            raise NotImplementedError(
                f"Could not parse function {func_obj.__name__}: {e}"
            )

        # 2. Evaluate arguments
        arg_vars = [self.visit(arg) for arg in node.args]

        if len(arg_vars) != len(func_def.args.args):
            raise NotImplementedError(
                f"Argument count mismatch for {func_obj.__name__}"
            )

        # 3. Generate unique suffix
        suffix = f"_{func_obj.__name__}_{self._get_unique_id()}"
        res_name = f"_res{suffix}"

        # Assume Int64 for now as match returns 0/1
        dtype = Scalar(PrimitiveType.Int64)
        self.builder.add_container(res_name, dtype, False)
        self.symbol_table[res_name] = dtype

        # 4. Rename variables
        class VariableRenamer(ast.NodeTransformer):
            # Builtins that should not be renamed
            BUILTINS = {
                "range",
                "len",
                "int",
                "float",
                "bool",
                "str",
                "list",
                "dict",
                "tuple",
                "set",
                "print",
                "abs",
                "min",
                "max",
                "sum",
                "enumerate",
                "zip",
                "map",
                "filter",
                "sorted",
                "reversed",
                "True",
                "False",
                "None",
            }

            def __init__(self, suffix, globals_dict):
                self.suffix = suffix
                self.globals_dict = globals_dict

            def visit_Name(self, node):
                # Don't rename builtins or globals
                if node.id in self.globals_dict or node.id in self.BUILTINS:
                    return node
                return ast.Name(id=f"{node.id}{self.suffix}", ctx=node.ctx)

            def visit_Return(self, node):
                if node.value:
                    val = self.visit(node.value)
                    return ast.Assign(
                        targets=[ast.Name(id=res_name, ctx=ast.Store())],
                        value=val,
                    )
                return node

        renamer = VariableRenamer(suffix, self.globals_dict)
        new_body = [renamer.visit(stmt) for stmt in func_def.body]

        # 5. Assign arguments to parameters
        param_assignments = []
        for arg_def, arg_val in zip(func_def.args.args, arg_vars):
            param_name = f"{arg_def.arg}{suffix}"

            # Infer type and create container
            if arg_val in self.symbol_table:
                self.symbol_table[param_name] = self.symbol_table[arg_val]
                self.builder.add_container(
                    param_name, self.symbol_table[arg_val], False
                )
                val_node = ast.Name(id=arg_val, ctx=ast.Load())
            elif self._is_int(arg_val):
                self.symbol_table[param_name] = Scalar(PrimitiveType.Int64)
                self.builder.add_container(
                    param_name, Scalar(PrimitiveType.Int64), False
                )
                val_node = ast.Constant(value=int(arg_val))
            else:
                # Assume float constant
                try:
                    val = float(arg_val)
                    self.symbol_table[param_name] = Scalar(PrimitiveType.Double)
                    self.builder.add_container(
                        param_name, Scalar(PrimitiveType.Double), False
                    )
                    val_node = ast.Constant(value=val)
                except ValueError:
                    # Fallback to Name, might fail later if not in symbol table
                    val_node = ast.Name(id=arg_val, ctx=ast.Load())

            assign = ast.Assign(
                targets=[ast.Name(id=param_name, ctx=ast.Store())], value=val_node
            )
            param_assignments.append(assign)

        final_body = param_assignments + new_body

        # 6. Visit new body using ASTParser
        from .ast_parser import ASTParser

        parser = ASTParser(
            self.builder,
            self.array_info,
            self.symbol_table,
            globals_dict=self.globals_dict,
            unique_counter_ref=self._unique_counter_ref,
        )

        for stmt in final_body:
            parser.visit(stmt)

        return res_name

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            return self._handle_numpy_matmul_op(node.left, node.right)

        left = self.visit(node.left)
        op = self.visit(node.op)
        right = self.visit(node.right)

        # Check if left or right are arrays
        left_is_array = left in self.array_info
        right_is_array = right in self.array_info

        if left_is_array or right_is_array:
            op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div", "**": "pow"}
            if op in op_map:
                return self._handle_array_binary_op(op_map[op], left, right)
            else:
                raise NotImplementedError(f"Array operation {op} not supported")

        tmp_name = f"_tmp_{self._get_unique_id()}"

        dtype = Scalar(PrimitiveType.Double)  # Default

        left_is_int = self._is_int(left)
        right_is_int = self._is_int(right)

        if left_is_int and right_is_int and op not in ["/", "**"]:
            dtype = Scalar(PrimitiveType.Int64)

        self.builder.add_container(tmp_name, dtype, False)
        self.symbol_table[tmp_name] = dtype

        real_left = left
        real_right = right

        if dtype.primitive_type == PrimitiveType.Double:
            if left_is_int:
                left_cast = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(
                    left_cast, Scalar(PrimitiveType.Double), False
                )
                self.symbol_table[left_cast] = Scalar(PrimitiveType.Double)

                c_block = self.builder.add_block()
                t_src, src_sub = self._add_read(c_block, left)
                t_dst = self.builder.add_access(c_block, left_cast)
                t_task = self.builder.add_tasklet(
                    c_block, TaskletCode.assign, ["_in"], ["_out"]
                )
                self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

                real_left = left_cast

            if right_is_int:
                right_cast = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(
                    right_cast, Scalar(PrimitiveType.Double), False
                )
                self.symbol_table[right_cast] = Scalar(PrimitiveType.Double)

                c_block = self.builder.add_block()
                t_src, src_sub = self._add_read(c_block, right)
                t_dst = self.builder.add_access(c_block, right_cast)
                t_task = self.builder.add_tasklet(
                    c_block, TaskletCode.assign, ["_in"], ["_out"]
                )
                self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

                real_right = right_cast

        # Special cases
        if op == "**":
            block = self.builder.add_block()
            t_left, left_sub = self._add_read(block, real_left)
            t_right, right_sub = self._add_read(block, real_right)
            t_out = self.builder.add_access(block, tmp_name)

            t_task = self.builder.add_cmath(block, CMathFunction.pow)
            self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
            self.builder.add_memlet(block, t_right, "void", t_task, "_in2", right_sub)
            self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")

            return tmp_name
        elif op == "%":
            block = self.builder.add_block()
            t_left, left_sub = self._add_read(block, real_left)
            t_right, right_sub = self._add_read(block, real_right)
            t_out = self.builder.add_access(block, tmp_name)

            if dtype.primitive_type == PrimitiveType.Int64:
                # Implement ((a % b) + b) % b to match Python's modulo behavior

                # 1. rem1 = a % b
                t_rem1 = self.builder.add_tasklet(
                    block, TaskletCode.int_srem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_left, "void", t_rem1, "_in1", left_sub)
                self.builder.add_memlet(
                    block, t_right, "void", t_rem1, "_in2", right_sub
                )

                rem1_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(rem1_name, dtype, False)
                t_rem1_out = self.builder.add_access(block, rem1_name)
                self.builder.add_memlet(block, t_rem1, "_out", t_rem1_out, "void", "")

                # 2. add = rem1 + b
                t_add = self.builder.add_tasklet(
                    block, TaskletCode.int_add, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_rem1_out, "void", t_add, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_add, "_in2", right_sub
                )

                add_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(add_name, dtype, False)
                t_add_out = self.builder.add_access(block, add_name)
                self.builder.add_memlet(block, t_add, "_out", t_add_out, "void", "")

                # 3. res = add % b
                t_rem2 = self.builder.add_tasklet(
                    block, TaskletCode.int_srem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_add_out, "void", t_rem2, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_rem2, "_in2", right_sub
                )
                self.builder.add_memlet(block, t_rem2, "_out", t_out, "void", "")

                return tmp_name
            else:
                # Python's floored modulo: a % b = a - floor(a / b) * b
                # This differs from fmod which uses trunc instead of floor
                # Implement as: fmod(fmod(a, b) + b, b) to handle negative values

                # 1. rem1 = fmod(a, b)
                t_rem1 = self.builder.add_tasklet(
                    block, TaskletCode.fp_rem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_left, "void", t_rem1, "_in1", left_sub)
                self.builder.add_memlet(
                    block, t_right, "void", t_rem1, "_in2", right_sub
                )

                rem1_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(rem1_name, dtype, False)
                t_rem1_out = self.builder.add_access(block, rem1_name)
                self.builder.add_memlet(block, t_rem1, "_out", t_rem1_out, "void", "")

                # 2. add = rem1 + b
                t_add = self.builder.add_tasklet(
                    block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_rem1_out, "void", t_add, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_add, "_in2", right_sub
                )

                add_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(add_name, dtype, False)
                t_add_out = self.builder.add_access(block, add_name)
                self.builder.add_memlet(block, t_add, "_out", t_add_out, "void", "")

                # 3. res = fmod(add, b)
                t_rem2 = self.builder.add_tasklet(
                    block, TaskletCode.fp_rem, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_add_out, "void", t_rem2, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_rem2, "_in2", right_sub
                )
                self.builder.add_memlet(block, t_rem2, "_out", t_out, "void", "")

                return tmp_name

        tasklet_code = None
        if dtype.primitive_type == PrimitiveType.Int64:
            if op == "+":
                tasklet_code = TaskletCode.int_add
            elif op == "-":
                tasklet_code = TaskletCode.int_sub
            elif op == "*":
                tasklet_code = TaskletCode.int_mul
            elif op == "/":
                tasklet_code = TaskletCode.int_sdiv
            elif op == "//":
                tasklet_code = TaskletCode.int_sdiv
            elif op == "|":
                tasklet_code = TaskletCode.int_or
            elif op == "^":
                tasklet_code = TaskletCode.int_xor
        else:
            if op == "+":
                tasklet_code = TaskletCode.fp_add
            elif op == "-":
                tasklet_code = TaskletCode.fp_sub
            elif op == "*":
                tasklet_code = TaskletCode.fp_mul
            elif op == "/":
                tasklet_code = TaskletCode.fp_div
            elif op == "//":
                tasklet_code = TaskletCode.fp_div
            else:
                raise NotImplementedError(f"Operation {op} not supported for floats")

        block = self.builder.add_block()
        t_left, left_sub = self._add_read(block, real_left)
        t_right, right_sub = self._add_read(block, real_right)
        t_out = self.builder.add_access(block, tmp_name)

        t_task = self.builder.add_tasklet(
            block, tasklet_code, ["_in1", "_in2"], ["_out"]
        )

        self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
        self.builder.add_memlet(block, t_right, "void", t_task, "_in2", right_sub)
        self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")

        return tmp_name

    def _add_assign_constant(self, target_name, value_str, dtype):
        block = self.builder.add_block()
        t_const = self.builder.add_constant(block, value_str, dtype)
        t_dst = self.builder.add_access(block, target_name)
        t_task = self.builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
        self.builder.add_memlet(block, t_const, "void", t_task, "_in", "")
        self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

    def visit_BoolOp(self, node):
        op = self.visit(node.op)
        values = [f"({self.visit(v)} != 0)" for v in node.values]
        expr_str = f"{f' {op} '.join(values)}"

        tmp_name = f"_tmp_{self._get_unique_id()}"
        dtype = Scalar(PrimitiveType.Bool)
        self.builder.add_container(tmp_name, dtype, False)

        # Use control flow to assign boolean value
        self.builder.begin_if(expr_str)
        self._add_assign_constant(tmp_name, "true", dtype)
        self.builder.begin_else()
        self._add_assign_constant(tmp_name, "false", dtype)
        self.builder.end_if()

        self.symbol_table[tmp_name] = dtype
        return tmp_name

    def visit_Compare(self, node):
        left = self.visit(node.left)
        if len(node.ops) > 1:
            raise NotImplementedError("Chained comparisons not supported yet")

        op = self.visit(node.ops[0])
        right = self.visit(node.comparators[0])

        # Check if this is an array comparison
        left_is_array = left in self.array_info
        right_is_array = right in self.array_info

        if left_is_array or right_is_array:
            # Handle array comparison - return boolean array
            return self._handle_array_compare(
                left, op, right, left_is_array, right_is_array
            )

        # Scalar comparison
        expr_str = f"{left} {op} {right}"

        tmp_name = f"_tmp_{self._get_unique_id()}"
        dtype = Scalar(PrimitiveType.Bool)
        self.builder.add_container(tmp_name, dtype, False)

        # Use control flow to assign boolean value
        self.builder.begin_if(expr_str)
        self.builder.add_transition(tmp_name, "true")
        self.builder.begin_else()
        self.builder.add_transition(tmp_name, "false")
        self.builder.end_if()

        self.symbol_table[tmp_name] = dtype
        return tmp_name

    def visit_UnaryOp(self, node):
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return f"-{node.operand.value}"

        op = self.visit(node.op)
        operand = self.visit(node.operand)

        # Check if operand is an array - handle as array operation
        if operand in self.array_info and op == "-":
            return self._handle_array_negate(operand)

        tmp_name = f"_tmp_{self._get_unique_id()}"
        dtype = Scalar(PrimitiveType.Double)
        if operand in self.symbol_table:
            dtype = self.symbol_table[operand]
            # If it's a pointer (array), get the element type
            if isinstance(dtype, Pointer) and dtype.has_pointee_type():
                dtype = dtype.pointee_type
        elif self._is_int(operand):
            dtype = Scalar(PrimitiveType.Int64)
        elif isinstance(node.op, ast.Not):
            dtype = Scalar(PrimitiveType.Bool)

        self.builder.add_container(tmp_name, dtype, False)
        self.symbol_table[tmp_name] = dtype

        block = self.builder.add_block()
        t_src, src_sub = self._add_read(block, operand)
        t_dst = self.builder.add_access(block, tmp_name)

        if isinstance(node.op, ast.Not):
            t_const = self.builder.add_constant(
                block, "true", Scalar(PrimitiveType.Bool)
            )
            t_task = self.builder.add_tasklet(
                block, TaskletCode.int_xor, ["_in1", "_in2"], ["_out"]
            )
            self.builder.add_memlet(block, t_src, "void", t_task, "_in1", src_sub)
            self.builder.add_memlet(block, t_const, "void", t_task, "_in2", "")
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

        elif op == "-":
            if dtype.primitive_type == PrimitiveType.Int64:
                t_const = self.builder.add_constant(block, "0", dtype)
                t_task = self.builder.add_tasklet(
                    block, TaskletCode.int_sub, ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_const, "void", t_task, "_in1", "")
                self.builder.add_memlet(block, t_src, "void", t_task, "_in2", src_sub)
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
            else:
                t_task = self.builder.add_tasklet(
                    block, TaskletCode.fp_neg, ["_in"], ["_out"]
                )
                self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
        else:
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

        return tmp_name

    def _handle_array_negate(self, operand):
        """Handle negation of an array operand (-arr)."""
        shape = self.array_info[operand]["shapes"]
        dtype = self._get_dtype(operand)

        # Create output array
        tmp_name = self._create_array_temp(shape, dtype)

        # Use elementwise binary op: 0 - arr
        # First create a zero constant
        zero_name = f"_tmp_{self._get_unique_id()}"
        self.builder.add_container(zero_name, dtype, False)
        self.symbol_table[zero_name] = dtype

        zero_block = self.builder.add_block()
        t_const = self.builder.add_constant(
            zero_block,
            "0.0" if dtype.primitive_type == PrimitiveType.Double else "0",
            dtype,
        )
        t_zero = self.builder.add_access(zero_block, zero_name)
        t_assign = self.builder.add_tasklet(
            zero_block, TaskletCode.assign, ["_in"], ["_out"]
        )
        self.builder.add_memlet(zero_block, t_const, "void", t_assign, "_in", "")
        self.builder.add_memlet(zero_block, t_assign, "_out", t_zero, "void", "")

        # Now subtract: tmp = 0 - operand (broadcast scalar subtraction)
        self.builder.add_elementwise_op("sub", zero_name, operand, tmp_name, shape)

        return tmp_name

    def _handle_array_compare(self, left, op, right, left_is_array, right_is_array):
        """Handle elementwise comparison of arrays, returning a boolean array.

        Supports: arr > 0, arr < scalar, arr1 > arr2, etc.
        """
        # Determine shape from the array operand
        if left_is_array:
            shape = self.array_info[left]["shapes"]
            arr_name = left
        else:
            shape = self.array_info[right]["shapes"]
            arr_name = right

        # Determine if we need integer or floating point comparison
        # based on the array element type
        use_int_cmp = False
        arr_dtype = self._get_dtype(arr_name)
        if arr_dtype.primitive_type in (PrimitiveType.Int32, PrimitiveType.Int64):
            use_int_cmp = True

        # Create output boolean array
        dtype = Scalar(PrimitiveType.Bool)
        tmp_name = self._create_array_temp(shape, dtype)

        # Map comparison operators to tasklet codes
        if use_int_cmp:
            cmp_ops = {
                ">": TaskletCode.int_sgt,
                ">=": TaskletCode.int_sge,
                "<": TaskletCode.int_slt,
                "<=": TaskletCode.int_sle,
                "==": TaskletCode.int_eq,
                "!=": TaskletCode.int_ne,
            }
        else:
            # Floating point ordered comparisons
            cmp_ops = {
                ">": TaskletCode.fp_ogt,
                ">=": TaskletCode.fp_oge,
                "<": TaskletCode.fp_olt,
                "<=": TaskletCode.fp_ole,
                "==": TaskletCode.fp_oeq,
                "!=": TaskletCode.fp_one,
            }

        if op not in cmp_ops:
            raise NotImplementedError(
                f"Comparison operator {op} not supported for arrays"
            )

        tasklet_code = cmp_ops[op]

        # For scalar operand, we may need to convert integer to float
        # Create a float constant if needed
        scalar_name = None
        if not left_is_array:
            scalar_name = left
        elif not right_is_array:
            scalar_name = right

        if scalar_name is not None and not use_int_cmp:
            # Check if scalar is an integer literal and convert to float
            if self._is_int(scalar_name):
                # Create a float constant
                float_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(
                    float_name, Scalar(PrimitiveType.Double), False
                )
                self.symbol_table[float_name] = Scalar(PrimitiveType.Double)

                block_conv = self.builder.add_block()
                t_const = self.builder.add_constant(
                    block_conv, f"{scalar_name}.0", Scalar(PrimitiveType.Double)
                )
                t_float = self.builder.add_access(block_conv, float_name)
                t_assign = self.builder.add_tasklet(
                    block_conv, TaskletCode.assign, ["_in"], ["_out"]
                )
                self.builder.add_memlet(
                    block_conv, t_const, "void", t_assign, "_in", ""
                )
                self.builder.add_memlet(
                    block_conv, t_assign, "_out", t_float, "void", ""
                )

                # Replace the scalar name with the converted float
                if not left_is_array:
                    left = float_name
                else:
                    right = float_name

        # Generate nested loops
        loop_vars = []
        for i, dim in enumerate(shape):
            loop_var = f"_cmp_i{i}_{self._get_unique_id()}"
            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)
            loop_vars.append(loop_var)
            self.builder.begin_for(loop_var, "0", str(dim), "1")

        # Compute linear index for array access
        linear_idx = self._compute_linear_index(loop_vars, shape, tmp_name, len(shape))

        # Create comparison block
        block = self.builder.add_block()

        # Read left operand
        if left_is_array:
            t_left = self.builder.add_access(block, left)
            left_sub = linear_idx
        else:
            t_left, left_sub = self._add_read(block, left)

        # Read right operand
        if right_is_array:
            t_right = self.builder.add_access(block, right)
            right_sub = linear_idx
        else:
            t_right, right_sub = self._add_read(block, right)

        # Output access
        t_out = self.builder.add_access(block, tmp_name)

        # Create tasklet for comparison
        t_task = self.builder.add_tasklet(
            block, tasklet_code, ["_in1", "_in2"], ["_out"]
        )

        # Connect memlets
        self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
        self.builder.add_memlet(block, t_right, "void", t_task, "_in2", right_sub)
        self.builder.add_memlet(block, t_task, "_out", t_out, "void", linear_idx)

        # Close loops
        for _ in loop_vars:
            self.builder.end_for()

        return tmp_name

    def _parse_array_arg(self, node, simple_visitor):
        if isinstance(node, ast.Name):
            if node.id in self.array_info:
                return node.id, [], self.array_info[node.id]["shapes"]
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.array_info:
                name = node.value.id
                ndim = self.array_info[name]["ndim"]

                indices = []
                if isinstance(node.slice, ast.Tuple):
                    indices = list(node.slice.elts)
                else:
                    indices = [node.slice]

                while len(indices) < ndim:
                    indices.append(ast.Slice(lower=None, upper=None, step=None))

                start_indices = []
                slice_shape = []

                for i, idx in enumerate(indices):
                    if isinstance(idx, ast.Slice):
                        start = "0"
                        if idx.lower:
                            start = simple_visitor.visit(idx.lower)
                        start_indices.append(start)

                        shapes = self.array_info[name]["shapes"]
                        dim_size = (
                            shapes[i] if i < len(shapes) else f"_{name}_shape_{i}"
                        )
                        stop = dim_size
                        if idx.upper:
                            stop = simple_visitor.visit(idx.upper)

                        size = f"({stop} - {start})"
                        slice_shape.append(size)
                    else:
                        val = simple_visitor.visit(idx)
                        start_indices.append(val)

                shapes = self.array_info[name]["shapes"]
                linear_index = ""
                for i in range(ndim):
                    term = start_indices[i]
                    for j in range(i + 1, ndim):
                        shape_val = shapes[j] if j < len(shapes) else None
                        shape_sym = (
                            shape_val if shape_val is not None else f"_{name}_shape_{j}"
                        )
                        term = f"({term} * {shape_sym})"

                    if i == 0:
                        linear_index = term
                    else:
                        linear_index = f"({linear_index} + {term})"

                return name, [linear_index], slice_shape

        return None, None, None

    def visit_Attribute(self, node):
        if node.attr == "shape":
            if isinstance(node.value, ast.Name) and node.value.id in self.array_info:
                return f"_shape_proxy_{node.value.id}"

        if isinstance(node.value, ast.Name) and node.value.id == "math":
            val = ""
            if node.attr == "pi":
                val = "M_PI"
            elif node.attr == "e":
                val = "M_E"

            if val:
                tmp_name = f"_tmp_{self._get_unique_id()}"
                dtype = Scalar(PrimitiveType.Double)
                self.builder.add_container(tmp_name, dtype, False)
                self.symbol_table[tmp_name] = dtype
                self._add_assign_constant(tmp_name, val, dtype)
                return tmp_name

        # Handle class member access (e.g., obj.x, obj.y)
        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            attr_name = node.attr

            # Check if the object is a class instance (has a Structure type)
            if obj_name in self.symbol_table:
                obj_type = self.symbol_table[obj_name]
                if isinstance(obj_type, Pointer) and obj_type.has_pointee_type():
                    pointee_type = obj_type.pointee_type
                    if isinstance(pointee_type, Structure):
                        struct_name = pointee_type.name

                        # Look up member index and type from structure info
                        if (
                            struct_name in self.structure_member_info
                            and attr_name in self.structure_member_info[struct_name]
                        ):
                            member_index, member_type = self.structure_member_info[
                                struct_name
                            ][attr_name]
                        else:
                            # This should not happen if structure was registered properly
                            raise RuntimeError(
                                f"Member '{attr_name}' not found in structure '{struct_name}'. "
                                f"Available members: {list(self.structure_member_info.get(struct_name, {}).keys())}"
                            )

                        # Generate a tasklet to access the member
                        tmp_name = f"_tmp_{self._get_unique_id()}"

                        self.builder.add_container(tmp_name, member_type, False)
                        self.symbol_table[tmp_name] = member_type

                        # Create a tasklet that reads the member
                        block = self.builder.add_block()
                        obj_access = self.builder.add_access(block, obj_name)
                        tmp_access = self.builder.add_access(block, tmp_name)

                        # Use tasklet to pass through the value
                        # The actual member selection is done via the memlet subset
                        tasklet = self.builder.add_tasklet(
                            block, TaskletCode.assign, ["_in"], ["_out"]
                        )

                        # Use member index in the subset to select the correct member
                        subset = "0," + str(member_index)
                        self.builder.add_memlet(
                            block, obj_access, "", tasklet, "_in", subset
                        )
                        self.builder.add_memlet(block, tasklet, "_out", tmp_access, "")

                        return tmp_name

        raise NotImplementedError(f"Attribute access {node.attr} not supported")

    def _handle_expression_slicing(self, node, value_str, indices_nodes, shapes, ndim):
        """Handle slicing in expressions (e.g., arr[1:, :, k+1]).

        Creates a temporary array, generates loops to copy sliced data,
        and returns the temporary array name.
        """
        if not self.builder:
            raise ValueError("Builder required for expression slicing")

        # Determine element type from source array
        dtype = Scalar(PrimitiveType.Double)
        if value_str in self.symbol_table:
            t = self.symbol_table[value_str]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        # Analyze each dimension: is it a slice or an index?
        # For slices, compute the resulting shape dimension
        # For indices, that dimension is collapsed
        result_shapes = []  # Shape of the resulting array (for SDFG)
        result_shapes_runtime = []  # Shape expressions for runtime evaluation
        slice_info = []  # List of (dim_idx, start_str, stop_str, step_str) for slices
        index_info = []  # List of (dim_idx, index_str) for point indices

        for i, idx in enumerate(indices_nodes):
            shape_val = shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"

            if isinstance(idx, ast.Slice):
                # Parse slice bounds - check for indirect access patterns
                start_str = "0"
                start_str_runtime = "0"  # For runtime shape evaluation
                if idx.lower is not None:
                    # Check if lower bound contains indirect array access
                    if self._contains_indirect_access(idx.lower):
                        start_str, start_str_runtime = (
                            self._materialize_indirect_access(
                                idx.lower, return_original_expr=True
                            )
                        )
                    else:
                        start_str = self.visit(idx.lower)
                        start_str_runtime = start_str
                    # Handle negative indices
                    if isinstance(start_str, str) and (
                        start_str.startswith("-") or start_str.startswith("(-")
                    ):
                        start_str = f"({shape_val} + {start_str})"
                        start_str_runtime = f"({shape_val} + {start_str_runtime})"

                stop_str = str(shape_val)
                stop_str_runtime = str(shape_val)
                if idx.upper is not None:
                    # Check if upper bound contains indirect array access
                    if self._contains_indirect_access(idx.upper):
                        stop_str, stop_str_runtime = self._materialize_indirect_access(
                            idx.upper, return_original_expr=True
                        )
                    else:
                        stop_str = self.visit(idx.upper)
                        stop_str_runtime = stop_str
                    # Handle negative indices
                    if isinstance(stop_str, str) and (
                        stop_str.startswith("-") or stop_str.startswith("(-")
                    ):
                        stop_str = f"({shape_val} + {stop_str})"
                        stop_str_runtime = f"({shape_val} + {stop_str_runtime})"

                step_str = "1"
                if idx.step is not None:
                    step_str = self.visit(idx.step)

                # Compute the size of this dimension in the result
                dim_size = f"({stop_str} - {start_str})"
                dim_size_runtime = f"({stop_str_runtime} - {start_str_runtime})"
                result_shapes.append(dim_size)
                result_shapes_runtime.append(dim_size_runtime)
                slice_info.append((i, start_str, stop_str, step_str))
            else:
                # Point index - dimension is collapsed
                # Check for indirect array access in the index
                if self._contains_indirect_access(idx):
                    index_str = self._materialize_indirect_access(idx)
                else:
                    index_str = self.visit(idx)
                # Handle negative indices
                if isinstance(index_str, str) and (
                    index_str.startswith("-") or index_str.startswith("(-")
                ):
                    index_str = f"({shape_val} + {index_str})"
                index_info.append((i, index_str))

        # Create temporary array for the result
        tmp_name = self._get_temp_name("_slice_tmp_")
        result_ndim = len(result_shapes)

        if result_ndim == 0:
            # All dimensions indexed - result is a scalar
            self.builder.add_container(tmp_name, dtype, False)
            self.symbol_table[tmp_name] = dtype
        else:
            # Result is an array - use _create_array_temp to handle allocation
            # Calculate size for malloc - use SDFG symbolic shapes
            size_str = "1"
            for dim in result_shapes:
                size_str = f"({size_str} * {dim})"

            element_size = self.builder.get_sizeof(dtype)
            total_size = f"({size_str} * {element_size})"

            # Create pointer
            ptr_type = Pointer(dtype)
            self.builder.add_container(tmp_name, ptr_type, False)
            self.symbol_table[tmp_name] = ptr_type
            # Store both SDFG shapes (for compilation) and runtime shapes (for evaluation)
            # The "shapes" field uses SDFG symbolic variables for malloc sizing
            # The "shapes_runtime" field uses original expressions for Python runtime evaluation
            self.array_info[tmp_name] = {
                "ndim": result_ndim,
                "shapes": result_shapes,  # Uses materialized variables for SDFG
                "shapes_runtime": result_shapes_runtime,  # Uses original expressions for runtime
            }

            # Malloc for the temporary array
            debug_info = DebugInfo()
            block_alloc = self.builder.add_block(debug_info)
            t_malloc = self.builder.add_malloc(block_alloc, total_size)
            t_ptr = self.builder.add_access(block_alloc, tmp_name, debug_info)
            self.builder.add_memlet(
                block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type, debug_info
            )

        # Generate loops to copy the sliced data
        loop_vars = []
        debug_info = DebugInfo()

        for dim_idx, (orig_dim, start_str, stop_str, step_str) in enumerate(slice_info):
            loop_var = f"_slice_loop_{dim_idx}_{self._get_unique_id()}"
            loop_vars.append((loop_var, orig_dim, start_str, step_str))

            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

            # Loop from 0 to (stop - start)
            count_str = f"({stop_str} - {start_str})"
            self.builder.begin_for(loop_var, "0", count_str, "1", debug_info)

        # Build source and destination indices
        src_indices = [""] * ndim
        dst_indices = []

        # Fill in point indices for source
        for orig_dim, index_str in index_info:
            src_indices[orig_dim] = index_str

        # Fill in slice indices for source and build destination indices
        for loop_var, orig_dim, start_str, step_str in loop_vars:
            if step_str == "1":
                src_indices[orig_dim] = f"({start_str} + {loop_var})"
            else:
                src_indices[orig_dim] = f"({start_str} + {loop_var} * {step_str})"
            dst_indices.append(loop_var)

        # Compute linear indices
        src_linear = self._compute_linear_index(src_indices, shapes, value_str, ndim)
        if result_ndim > 0:
            dst_linear = self._compute_linear_index(
                dst_indices, result_shapes, tmp_name, result_ndim
            )
        else:
            dst_linear = "0"

        # Create the copy block
        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, value_str, debug_info)
        t_dst = self.builder.add_access(block, tmp_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_linear, None, debug_info
        )
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", dst_linear, None, debug_info
        )

        # Close all loops
        for _ in loop_vars:
            self.builder.end_for()

        return tmp_name

    def _compute_linear_index(self, indices, shapes, array_name, ndim):
        """Compute linear index from multi-dimensional indices."""
        if ndim == 0:
            return "0"

        linear_index = ""
        for i in range(ndim):
            term = str(indices[i])
            for j in range(i + 1, ndim):
                shape_val = shapes[j] if j < len(shapes) else f"_{array_name}_shape_{j}"
                term = f"(({term}) * {shape_val})"

            if i == 0:
                linear_index = term
            else:
                linear_index = f"({linear_index} + {term})"

        return linear_index

    def _is_array_index(self, node):
        """Check if a node represents an array that could be used as an index (gather).

        Returns True if the node is a Name referring to an array in array_info.
        """
        if isinstance(node, ast.Name):
            return node.id in self.array_info
        return False

    def _handle_gather(self, value_str, index_node, debug_info=None):
        """Handle gather operation: x[indices] where indices is an array.

        Creates a temporary array and generates a loop to gather elements
        from the source array using the index array.

        This is the canonical SDFG pattern for gather operations:
        - Create a loop over the index array
        - Load the index value using a tasklet+memlets
        - Use that index in the memlet subset for the source array
        """
        if debug_info is None:
            debug_info = DebugInfo()

        # Get the index array name
        if isinstance(index_node, ast.Name):
            idx_array_name = index_node.id
        else:
            # Visit the index to get its name (handles slices like cols)
            idx_array_name = self.visit(index_node)

        if idx_array_name not in self.array_info:
            raise ValueError(f"Gather index must be an array, got {idx_array_name}")

        # Get shapes
        idx_shapes = self.array_info[idx_array_name].get("shapes", [])
        src_ndim = self.array_info[value_str]["ndim"]
        idx_ndim = self.array_info[idx_array_name]["ndim"]

        if idx_ndim != 1:
            raise NotImplementedError("Only 1D index arrays supported for gather")

        # Result array has same shape as index array
        result_shape = idx_shapes[0] if idx_shapes else f"_{idx_array_name}_shape_0"

        # Determine element type from source array
        dtype = Scalar(PrimitiveType.Double)
        if value_str in self.symbol_table:
            t = self.symbol_table[value_str]
            if isinstance(t, Pointer) and t.has_pointee_type():
                dtype = t.pointee_type

        # Determine index type from index array
        idx_dtype = Scalar(PrimitiveType.Int64)
        if idx_array_name in self.symbol_table:
            t = self.symbol_table[idx_array_name]
            if isinstance(t, Pointer) and t.has_pointee_type():
                idx_dtype = t.pointee_type

        # Create result array
        tmp_name = self._get_temp_name("_gather_")

        # Calculate size for malloc
        element_size = self.builder.get_sizeof(dtype)
        total_size = f"({result_shape} * {element_size})"

        # Create pointer for result
        ptr_type = Pointer(dtype)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.symbol_table[tmp_name] = ptr_type
        self.array_info[tmp_name] = {"ndim": 1, "shapes": [result_shape]}

        # Malloc for the result array
        block_alloc = self.builder.add_block(debug_info)
        t_malloc = self.builder.add_malloc(block_alloc, total_size)
        t_ptr = self.builder.add_access(block_alloc, tmp_name, debug_info)
        self.builder.add_memlet(
            block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type, debug_info
        )

        # Create loop variable
        loop_var = f"_gather_i_{self._get_unique_id()}"
        self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
        self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

        # Create variable to hold the loaded index
        idx_var = f"_gather_idx_{self._get_unique_id()}"
        self.builder.add_container(idx_var, idx_dtype, False)
        self.symbol_table[idx_var] = idx_dtype

        # Begin loop
        self.builder.begin_for(loop_var, "0", str(result_shape), "1", debug_info)

        # Block 1: Load the index from index array using tasklet+memlets
        block_load_idx = self.builder.add_block(debug_info)
        idx_arr_access = self.builder.add_access(
            block_load_idx, idx_array_name, debug_info
        )
        idx_var_access = self.builder.add_access(block_load_idx, idx_var, debug_info)
        tasklet_load = self.builder.add_tasklet(
            block_load_idx, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        self.builder.add_memlet(
            block_load_idx,
            idx_arr_access,
            "void",
            tasklet_load,
            "_in",
            loop_var,
            None,
            debug_info,
        )
        self.builder.add_memlet(
            block_load_idx,
            tasklet_load,
            "_out",
            idx_var_access,
            "void",
            "",
            None,
            debug_info,
        )

        # Block 2: Use the loaded index to gather from source array
        block_gather = self.builder.add_block(debug_info)
        src_access = self.builder.add_access(block_gather, value_str, debug_info)
        dst_access = self.builder.add_access(block_gather, tmp_name, debug_info)
        tasklet_gather = self.builder.add_tasklet(
            block_gather, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        # Use the symbolic variable name (idx_var) in the memlet subset - this is key!
        self.builder.add_memlet(
            block_gather,
            src_access,
            "void",
            tasklet_gather,
            "_in",
            idx_var,
            None,
            debug_info,
        )
        self.builder.add_memlet(
            block_gather,
            tasklet_gather,
            "_out",
            dst_access,
            "void",
            loop_var,
            None,
            debug_info,
        )

        # End loop
        self.builder.end_for()

        return tmp_name

    def visit_Subscript(self, node):
        value_str = self.visit(node.value)

        if value_str.startswith("_shape_proxy_"):
            array_name = value_str[len("_shape_proxy_") :]
            if isinstance(node.slice, ast.Constant):
                idx = node.slice.value
            elif isinstance(node.slice, ast.Index):
                idx = node.slice.value.value
            else:
                try:
                    idx = int(self.visit(node.slice))
                except:
                    raise NotImplementedError(
                        "Dynamic shape indexing not fully supported yet"
                    )

            if (
                array_name in self.array_info
                and "shapes" in self.array_info[array_name]
            ):
                return self.array_info[array_name]["shapes"][idx]

            return f"_{array_name}_shape_{idx}"

        if value_str in self.array_info:
            ndim = self.array_info[value_str]["ndim"]
            shapes = self.array_info[value_str].get("shapes", [])

            indices = []
            if isinstance(node.slice, ast.Tuple):
                indices_nodes = node.slice.elts
            else:
                indices_nodes = [node.slice]

            # Check if all indices are full slices (e.g., path[:] or path[:, :])
            # In this case, return just the array name since it's the full array
            all_full_slices = True
            for idx in indices_nodes:
                if isinstance(idx, ast.Slice):
                    # A full slice has no lower, upper bounds or only None
                    if idx.lower is not None or idx.upper is not None:
                        all_full_slices = False
                        break
                else:
                    all_full_slices = False
                    break

            # path[:] on an nD array returns the full array
            # So if we have a single full slice, it covers all dimensions
            if all_full_slices:
                # This is path[:] or path[:,:] - return the array name
                return value_str

            # Check if there are any slices in the indices
            has_slices = any(isinstance(idx, ast.Slice) for idx in indices_nodes)
            if has_slices:
                # Handle mixed slicing (e.g., arr[1:, :, k] or arr[:-1, :, k+1])
                return self._handle_expression_slicing(
                    node, value_str, indices_nodes, shapes, ndim
                )

            # Check for gather operation: x[indices_array] where indices_array is an array
            # This happens when we have a 1D source array and a 1D index array
            if len(indices_nodes) == 1 and self._is_array_index(indices_nodes[0]):
                if self.builder:
                    return self._handle_gather(value_str, indices_nodes[0])

            if isinstance(node.slice, ast.Tuple):
                indices = [self.visit(elt) for elt in node.slice.elts]
            else:
                indices = [self.visit(node.slice)]

            if len(indices) != ndim:
                raise ValueError(
                    f"Array {value_str} has {ndim} dimensions, but accessed with {len(indices)} indices"
                )

            # Normalize negative indices
            normalized_indices = []
            for i, idx_str in enumerate(indices):
                shape_val = shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"
                # Check if index is negative (starts with "-" or "(-")
                if isinstance(idx_str, str) and (
                    idx_str.startswith("-") or idx_str.startswith("(-")
                ):
                    # Normalize: size + negative_index
                    normalized_indices.append(f"({shape_val} + {idx_str})")
                else:
                    normalized_indices.append(idx_str)

            linear_index = ""
            for i in range(ndim):
                term = normalized_indices[i]
                for j in range(i + 1, ndim):
                    shape_val = shapes[j] if j < len(shapes) else None
                    shape_sym = (
                        shape_val
                        if shape_val is not None
                        else f"_{value_str}_shape_{j}"
                    )
                    term = f"(({term}) * {shape_sym})"

                if i == 0:
                    linear_index = term
                else:
                    linear_index = f"({linear_index} + {term})"

            access_str = f"{value_str}({linear_index})"

            if self.builder and isinstance(node.ctx, ast.Load):
                dtype = Scalar(PrimitiveType.Double)
                if value_str in self.symbol_table:
                    t = self.symbol_table[value_str]
                    if type(t).__name__ == "Array" and hasattr(t, "element_type"):
                        et = t.element_type
                        if callable(et):
                            et = et()
                        dtype = et
                    elif type(t).__name__ == "Pointer" and hasattr(t, "pointee_type"):
                        et = t.pointee_type
                        if callable(et):
                            et = et()
                        dtype = et

                tmp_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(tmp_name, dtype, False)

                block = self.builder.add_block()
                t_src = self.builder.add_access(block, value_str)
                t_dst = self.builder.add_access(block, tmp_name)
                t_task = self.builder.add_tasklet(
                    block, TaskletCode.assign, ["_in"], ["_out"]
                )

                self.builder.add_memlet(
                    block, t_src, "void", t_task, "_in", linear_index
                )
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

                self.symbol_table[tmp_name] = dtype
                return tmp_name

            return access_str

        slice_val = self.visit(node.slice)
        access_str = f"{value_str}({slice_val})"

        if (
            self.builder
            and isinstance(node.ctx, ast.Load)
            and value_str in self.array_info
        ):
            tmp_name = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(tmp_name, Scalar(PrimitiveType.Double), False)
            self.builder.add_assignment(tmp_name, access_str)
            self.symbol_table[tmp_name] = Scalar(PrimitiveType.Double)
            return tmp_name

        return access_str

    def visit_Add(self, node):
        return "+"

    def visit_Sub(self, node):
        return "-"

    def visit_Mult(self, node):
        return "*"

    def visit_Div(self, node):
        return "/"

    def visit_FloorDiv(self, node):
        return "//"

    def visit_Mod(self, node):
        return "%"

    def visit_Pow(self, node):
        return "**"

    def visit_Eq(self, node):
        return "=="

    def visit_NotEq(self, node):
        return "!="

    def visit_Lt(self, node):
        return "<"

    def visit_LtE(self, node):
        return "<="

    def visit_Gt(self, node):
        return ">"

    def visit_GtE(self, node):
        return ">="

    def visit_And(self, node):
        return "&"

    def visit_Or(self, node):
        return "|"

    def visit_BitAnd(self, node):
        return "&"

    def visit_BitOr(self, node):
        return "|"

    def visit_BitXor(self, node):
        return "^"

    def visit_Not(self, node):
        return "!"

    def visit_USub(self, node):
        return "-"

    def visit_UAdd(self, node):
        return "+"

    def visit_Invert(self, node):
        return "~"

    def _get_dtype(self, name):
        if name in self.symbol_table:
            t = self.symbol_table[name]
            if isinstance(t, Scalar):
                return t

            if hasattr(t, "pointee_type"):
                et = t.pointee_type
                if callable(et):
                    et = et()
                if isinstance(et, Scalar):
                    return et

            if hasattr(t, "element_type"):
                et = t.element_type
                if callable(et):
                    et = et()
                if isinstance(et, Scalar):
                    return et

        if self._is_int(name):
            return Scalar(PrimitiveType.Int64)

        return Scalar(PrimitiveType.Double)

    def _promote_dtypes(self, dtype_left, dtype_right):
        """Promote two dtypes following NumPy rules: float > int, wider > narrower."""
        # Priority order: Double > Float > Int64 > Int32
        priority = {
            PrimitiveType.Double: 4,
            PrimitiveType.Float: 3,
            PrimitiveType.Int64: 2,
            PrimitiveType.Int32: 1,
        }
        left_prio = priority.get(dtype_left.primitive_type, 0)
        right_prio = priority.get(dtype_right.primitive_type, 0)
        if left_prio >= right_prio:
            return dtype_left
        else:
            return dtype_right

    def _create_array_temp(
        self, shape, dtype, zero_init=False, ones_init=False, shapes_runtime=None
    ):
        tmp_name = f"_tmp_{self._get_unique_id()}"

        # Handle 0-dimensional arrays as scalars
        if not shape or (len(shape) == 0):
            # 0-D array is just a scalar
            self.builder.add_container(tmp_name, dtype, False)
            self.symbol_table[tmp_name] = dtype
            self.array_info[tmp_name] = {"ndim": 0, "shapes": []}

            if zero_init:
                self.builder.add_assignment(
                    tmp_name,
                    "0.0" if dtype.primitive_type == PrimitiveType.Double else "0",
                )
            elif ones_init:
                self.builder.add_assignment(
                    tmp_name,
                    "1.0" if dtype.primitive_type == PrimitiveType.Double else "1",
                )

            return tmp_name

        # Calculate size
        size_str = "1"
        for dim in shape:
            size_str = f"({size_str} * {dim})"

        element_size = self.builder.get_sizeof(dtype)
        total_size = f"({size_str} * {element_size})"

        # Create pointer
        ptr_type = Pointer(dtype)
        self.builder.add_container(tmp_name, ptr_type, False)
        self.symbol_table[tmp_name] = ptr_type
        array_info_entry = {"ndim": len(shape), "shapes": shape}
        if shapes_runtime is not None:
            array_info_entry["shapes_runtime"] = shapes_runtime
        self.array_info[tmp_name] = array_info_entry

        # Malloc
        block1 = self.builder.add_block()
        t_malloc = self.builder.add_malloc(block1, total_size)
        t_ptr1 = self.builder.add_access(block1, tmp_name)
        self.builder.add_memlet(block1, t_malloc, "_ret", t_ptr1, "void", "", ptr_type)

        if zero_init:
            block2 = self.builder.add_block()
            t_memset = self.builder.add_memset(block2, "0", total_size)
            t_ptr2 = self.builder.add_access(block2, tmp_name)
            self.builder.add_memlet(
                block2, t_memset, "_ptr", t_ptr2, "void", "", ptr_type
            )
        elif ones_init:
            # Initialize array with ones using a loop
            loop_var = f"_i_{self._get_unique_id()}"
            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

            self.builder.begin_for(loop_var, "0", size_str, "1")

            # Determine the value to set based on dtype
            val = "1.0"
            if dtype.primitive_type in [
                PrimitiveType.Int64,
                PrimitiveType.Int32,
                PrimitiveType.Int8,
                PrimitiveType.Int16,
                PrimitiveType.UInt64,
                PrimitiveType.UInt32,
                PrimitiveType.UInt8,
                PrimitiveType.UInt16,
            ]:
                val = "1"

            block_assign = self.builder.add_block()
            t_const = self.builder.add_constant(block_assign, val, dtype)
            t_arr = self.builder.add_access(block_assign, tmp_name)

            t_task = self.builder.add_tasklet(
                block_assign, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_assign, t_const, "void", t_task, "_in", "", dtype
            )
            self.builder.add_memlet(
                block_assign, t_task, "_out", t_arr, "void", loop_var
            )

            self.builder.end_for()

        return tmp_name

    def _handle_array_unary_op(self, op_type, operand):
        # Determine output shape
        shape = []
        if operand in self.array_info:
            shape = self.array_info[operand]["shapes"]

        # Determine dtype
        dtype = self._get_dtype(operand)

        # For 0-D arrays (scalars), use an intrinsic (CMathNode) instead of library node
        if not shape or len(shape) == 0:
            tmp_name = self._create_array_temp(shape, dtype)

            # Map op_type to C function names
            func_map = {
                "sqrt": CMathFunction.sqrt,
                "abs": CMathFunction.fabs,
                "absolute": CMathFunction.fabs,
                "exp": CMathFunction.exp,
                "tanh": CMathFunction.tanh,
            }

            block = self.builder.add_block()
            t_src = self.builder.add_access(block, operand)
            t_dst = self.builder.add_access(block, tmp_name)
            t_task = self.builder.add_cmath(block, func_map[op_type])

            # CMathNode uses _in1, _in2, etc for inputs and _out for output
            self.builder.add_memlet(block, t_src, "void", t_task, "_in1", "", dtype)
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "", dtype)

            return tmp_name

        tmp_name = self._create_array_temp(shape, dtype)

        # Add operation
        self.builder.add_elementwise_unary_op(op_type, operand, tmp_name, shape)

        return tmp_name

    def _handle_array_binary_op(self, op_type, left, right):
        # Determine output shape (handle broadcasting by picking the larger shape)
        left_shape = []
        right_shape = []
        if left in self.array_info:
            left_shape = self.array_info[left]["shapes"]
        if right in self.array_info:
            right_shape = self.array_info[right]["shapes"]
        # Pick the shape with more dimensions for broadcasting
        shape = left_shape if len(left_shape) >= len(right_shape) else right_shape

        # Determine dtype with promotion (float > int, wider > narrower)
        dtype_left = self._get_dtype(left)
        dtype_right = self._get_dtype(right)

        # Promote dtypes: Double > Float > Int64 > Int32
        dtype = self._promote_dtypes(dtype_left, dtype_right)

        # Cast scalar operands to the promoted dtype if needed
        real_left = left
        real_right = right

        # Helper to check if operand is a scalar (not an array)
        left_is_scalar = left not in self.array_info
        right_is_scalar = right not in self.array_info

        # Cast left operand if needed (scalar int to float)
        if left_is_scalar and dtype_left.primitive_type != dtype.primitive_type:
            left_cast = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(left_cast, dtype, False)
            self.symbol_table[left_cast] = dtype

            c_block = self.builder.add_block()
            t_src, src_sub = self._add_read(c_block, left)
            t_dst = self.builder.add_access(c_block, left_cast)
            t_task = self.builder.add_tasklet(
                c_block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

            real_left = left_cast

        # Cast right operand if needed (scalar int to float)
        if right_is_scalar and dtype_right.primitive_type != dtype.primitive_type:
            right_cast = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(right_cast, dtype, False)
            self.symbol_table[right_cast] = dtype

            c_block = self.builder.add_block()
            t_src, src_sub = self._add_read(c_block, right)
            t_dst = self.builder.add_access(c_block, right_cast)
            t_task = self.builder.add_tasklet(
                c_block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

            real_right = right_cast

        tmp_name = self._create_array_temp(shape, dtype)

        # Add operation with promoted dtype for implicit casting
        self.builder.add_elementwise_op(op_type, real_left, real_right, tmp_name, shape)

        return tmp_name

    def _shape_to_runtime_expr(self, shape_node):
        """Convert a shape expression AST node to a runtime-evaluable string.

        This converts the AST to a string expression that can be evaluated
        at runtime using only input arrays and shape symbols (_s0, _s1, etc.).
        It does NOT visit the node (which would create SDFG variables).
        """
        if isinstance(shape_node, ast.Constant):
            return str(shape_node.value)
        elif isinstance(shape_node, ast.Name):
            return shape_node.id
        elif isinstance(shape_node, ast.BinOp):
            left = self._shape_to_runtime_expr(shape_node.left)
            right = self._shape_to_runtime_expr(shape_node.right)
            op = self.visit(shape_node.op)
            return f"({left} {op} {right})"
        elif isinstance(shape_node, ast.UnaryOp):
            operand = self._shape_to_runtime_expr(shape_node.operand)
            if isinstance(shape_node.op, ast.USub):
                return f"(-{operand})"
            elif isinstance(shape_node.op, ast.UAdd):
                return operand
            else:
                # Fall back to visit for other unary ops
                return self.visit(shape_node)
        elif isinstance(shape_node, ast.Subscript):
            # Handle arr.shape[0] -> arr.shape[0] for runtime eval
            # or _shape_proxy_arr[0] -> _s<idx>
            val = shape_node.value
            if isinstance(val, ast.Attribute) and val.attr == "shape":
                # arr.shape[0] -> use the shape symbol
                if isinstance(val.value, ast.Name):
                    arr_name = val.value.id
                    if isinstance(shape_node.slice, ast.Constant):
                        idx = shape_node.slice.value
                        # Get the shape symbol for this array dimension
                        if arr_name in self.array_info:
                            shapes = self.array_info[arr_name].get("shapes", [])
                            if idx < len(shapes):
                                return shapes[idx]
                        return f"{arr_name}.shape[{idx}]"
            # Fall back to visit
            return self.visit(shape_node)
        elif isinstance(shape_node, ast.Tuple):
            return [self._shape_to_runtime_expr(elt) for elt in shape_node.elts]
        elif isinstance(shape_node, ast.List):
            return [self._shape_to_runtime_expr(elt) for elt in shape_node.elts]
        else:
            # Fall back to visit for complex expressions
            return self.visit(shape_node)

    def _handle_numpy_alloc(self, node, func_name):
        # Parse shape
        shape_arg = node.args[0]
        dims = []
        dims_runtime = []  # Runtime-evaluable shape expressions
        if isinstance(shape_arg, ast.Tuple):
            dims = [self.visit(elt) for elt in shape_arg.elts]
            dims_runtime = [self._shape_to_runtime_expr(elt) for elt in shape_arg.elts]
        elif isinstance(shape_arg, ast.List):
            dims = [self.visit(elt) for elt in shape_arg.elts]
            dims_runtime = [self._shape_to_runtime_expr(elt) for elt in shape_arg.elts]
        else:
            val = self.visit(shape_arg)
            runtime_val = self._shape_to_runtime_expr(shape_arg)
            if val.startswith("_shape_proxy_"):
                array_name = val[len("_shape_proxy_") :]
                if array_name in self.array_info:
                    dims = self.array_info[array_name]["shapes"]
                    dims_runtime = self.array_info[array_name].get(
                        "shapes_runtime", dims
                    )
                else:
                    dims = [val]
                    dims_runtime = [runtime_val]
            else:
                dims = [val]
                dims_runtime = [runtime_val]

        # Parse dtype
        dtype_arg = None
        if len(node.args) > 1:
            dtype_arg = node.args[1]

        for kw in node.keywords:
            if kw.arg == "dtype":
                dtype_arg = kw.value
                break

        element_type = self._map_numpy_dtype(dtype_arg)

        return self._create_array_temp(
            dims,
            element_type,
            zero_init=(func_name == "zeros"),
            ones_init=(func_name == "ones"),
            shapes_runtime=dims_runtime,
        )

    def _handle_numpy_empty_like(self, node, func_name):
        prototype_arg = node.args[0]
        prototype_name = self.visit(prototype_arg)

        # Parse shape from prototype
        dims = []
        if prototype_name in self.array_info:
            dims = self.array_info[prototype_name]["shapes"]

        # Parse dtype
        dtype_arg = None
        if len(node.args) > 1:
            dtype_arg = node.args[1]

        for kw in node.keywords:
            if kw.arg == "dtype":
                dtype_arg = kw.value
                break

        element_type = None
        if dtype_arg:
            element_type = self._map_numpy_dtype(dtype_arg)
        else:
            if prototype_name in self.symbol_table:
                sym_type = self.symbol_table[prototype_name]
                if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                    element_type = sym_type.pointee_type

        if element_type is None:
            element_type = Scalar(PrimitiveType.Double)

        return self._create_array_temp(
            dims,
            element_type,
            zero_init=False,
            ones_init=False,
        )

    def _handle_numpy_zeros_like(self, node, func_name):
        prototype_arg = node.args[0]
        prototype_name = self.visit(prototype_arg)

        # Parse shape from prototype
        dims = []
        if prototype_name in self.array_info:
            dims = self.array_info[prototype_name]["shapes"]

        # Parse dtype
        dtype_arg = None
        if len(node.args) > 1:
            dtype_arg = node.args[1]

        for kw in node.keywords:
            if kw.arg == "dtype":
                dtype_arg = kw.value
                break

        element_type = None
        if dtype_arg:
            element_type = self._map_numpy_dtype(dtype_arg)
        else:
            if prototype_name in self.symbol_table:
                sym_type = self.symbol_table[prototype_name]
                if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                    element_type = sym_type.pointee_type

        if element_type is None:
            element_type = Scalar(PrimitiveType.Double)

        return self._create_array_temp(
            dims,
            element_type,
            zero_init=True,
            ones_init=False,
        )

    def _handle_numpy_eye(self, node, func_name):
        # Parse N
        N_arg = node.args[0]
        N_str = self.visit(N_arg)

        # Parse M
        M_str = N_str
        if len(node.args) > 1:
            M_str = self.visit(node.args[1])

        # Parse k
        k_str = "0"
        if len(node.args) > 2:
            k_str = self.visit(node.args[2])

        # Check keywords for M, k, dtype
        dtype_arg = None
        for kw in node.keywords:
            if kw.arg == "M":
                M_str = self.visit(kw.value)
                if M_str == "None":
                    M_str = N_str
            elif kw.arg == "k":
                k_str = self.visit(kw.value)
            elif kw.arg == "dtype":
                dtype_arg = kw.value

        element_type = self._map_numpy_dtype(dtype_arg)

        ptr_name = self._create_array_temp([N_str, M_str], element_type, zero_init=True)

        # Loop to set diagonal
        loop_var = f"_i_{self._get_unique_id()}"
        if not self.builder.exists(loop_var):
            self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
            self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

        self.builder.begin_for(loop_var, "0", N_str, "1")

        # Condition: 0 <= i + k < M
        cond = f"(({loop_var} + {k_str}) >= 0) & (({loop_var} + {k_str}) < {M_str})"
        self.builder.begin_if(cond)

        # Assignment: A[i, i+k] = 1
        val = "1.0"
        if element_type.primitive_type in [
            PrimitiveType.Int64,
            PrimitiveType.Int32,
            PrimitiveType.Int8,
            PrimitiveType.Int16,
            PrimitiveType.UInt64,
            PrimitiveType.UInt32,
            PrimitiveType.UInt8,
            PrimitiveType.UInt16,
        ]:
            val = "1"

        block_assign = self.builder.add_block()
        t_const = self.builder.add_constant(block_assign, val, element_type)
        t_arr = self.builder.add_access(block_assign, ptr_name)
        flat_index = f"(({loop_var}) * ({M_str}) + ({loop_var}) + ({k_str}))"
        subset = flat_index

        t_task = self.builder.add_tasklet(
            block_assign, TaskletCode.assign, ["_in"], ["_out"]
        )
        self.builder.add_memlet(
            block_assign, t_const, "void", t_task, "_in", "", element_type
        )
        self.builder.add_memlet(block_assign, t_task, "_out", t_arr, "void", subset)

        self.builder.end_if()
        self.builder.end_for()

        return ptr_name

    def _handle_numpy_binary_op(self, node, func_name):
        args = [self.visit(arg) for arg in node.args]
        if len(args) != 2:
            raise NotImplementedError(
                f"Numpy function {func_name} requires 2 arguments"
            )

        op_map = {
            "add": "add",
            "subtract": "sub",
            "multiply": "mul",
            "divide": "div",
            "power": "pow",
            "minimum": "min",
            "maximum": "max",
        }
        return self._handle_array_binary_op(op_map[func_name], args[0], args[1])

    def _handle_numpy_where(self, node, func_name):
        """Handle np.where(condition, x, y) - elementwise ternary selection.

        Returns an array where elements are taken from x where condition is True,
        and from y where condition is False.
        """
        if len(node.args) != 3:
            raise NotImplementedError("np.where requires 3 arguments (condition, x, y)")

        # Visit all arguments
        cond_name = self.visit(node.args[0])
        x_name = self.visit(node.args[1])
        y_name = self.visit(node.args[2])

        # Determine output shape from the array arguments
        # Priority: condition > y > x (since x might be scalar 0)
        shape = []
        dtype = Scalar(PrimitiveType.Double)

        # Check condition shape
        if cond_name in self.array_info:
            shape = self.array_info[cond_name]["shapes"]

        # If condition is scalar, check y
        if not shape and y_name in self.array_info:
            shape = self.array_info[y_name]["shapes"]

        # If y is scalar, check x
        if not shape and x_name in self.array_info:
            shape = self.array_info[x_name]["shapes"]

        if not shape:
            raise NotImplementedError("np.where requires at least one array argument")

        # Determine dtype from y (since x might be scalar 0)
        if y_name in self.symbol_table:
            y_type = self.symbol_table[y_name]
            if isinstance(y_type, Pointer) and y_type.has_pointee_type():
                dtype = y_type.pointee_type
            elif isinstance(y_type, Scalar):
                dtype = y_type

        # Create output array
        tmp_name = self._create_array_temp(shape, dtype)

        # Generate nested loops for the shape
        loop_vars = []
        for i, dim in enumerate(shape):
            loop_var = f"_where_i{i}_{self._get_unique_id()}"
            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)
            loop_vars.append(loop_var)
            self.builder.begin_for(loop_var, "0", str(dim), "1")

        # Compute linear index
        linear_idx = self._compute_linear_index(loop_vars, shape, tmp_name, len(shape))

        # Read condition value
        cond_tmp = f"_where_cond_{self._get_unique_id()}"
        self.builder.add_container(cond_tmp, Scalar(PrimitiveType.Bool), False)
        self.symbol_table[cond_tmp] = Scalar(PrimitiveType.Bool)

        block_cond = self.builder.add_block()
        if cond_name in self.array_info:
            t_cond_arr = self.builder.add_access(block_cond, cond_name)
            t_cond_out = self.builder.add_access(block_cond, cond_tmp)
            t_cond_task = self.builder.add_tasklet(
                block_cond, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_cond, t_cond_arr, "void", t_cond_task, "_in", linear_idx
            )
            self.builder.add_memlet(
                block_cond, t_cond_task, "_out", t_cond_out, "void", ""
            )
        else:
            # Scalar condition - just use it directly
            t_cond_src, cond_sub = self._add_read(block_cond, cond_name)
            t_cond_out = self.builder.add_access(block_cond, cond_tmp)
            t_cond_task = self.builder.add_tasklet(
                block_cond, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_cond, t_cond_src, "void", t_cond_task, "_in", cond_sub
            )
            self.builder.add_memlet(
                block_cond, t_cond_task, "_out", t_cond_out, "void", ""
            )

        # If-else based on condition
        self.builder.begin_if(f"{cond_tmp} == true")

        # True branch: assign x
        block_true = self.builder.add_block()
        t_out_true = self.builder.add_access(block_true, tmp_name)
        if x_name in self.array_info:
            # x is an array
            t_x = self.builder.add_access(block_true, x_name)
            t_task_true = self.builder.add_tasklet(
                block_true, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_true, t_x, "void", t_task_true, "_in", linear_idx
            )
        else:
            # x is a scalar
            t_x, x_sub = self._add_read(block_true, x_name)
            t_task_true = self.builder.add_tasklet(
                block_true, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(block_true, t_x, "void", t_task_true, "_in", x_sub)
        self.builder.add_memlet(
            block_true, t_task_true, "_out", t_out_true, "void", linear_idx
        )

        self.builder.begin_else()

        # False branch: assign y
        block_false = self.builder.add_block()
        t_out_false = self.builder.add_access(block_false, tmp_name)
        if y_name in self.array_info:
            # y is an array
            t_y = self.builder.add_access(block_false, y_name)
            t_task_false = self.builder.add_tasklet(
                block_false, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_false, t_y, "void", t_task_false, "_in", linear_idx
            )
        else:
            # y is a scalar
            t_y, y_sub = self._add_read(block_false, y_name)
            t_task_false = self.builder.add_tasklet(
                block_false, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_false, t_y, "void", t_task_false, "_in", y_sub
            )
        self.builder.add_memlet(
            block_false, t_task_false, "_out", t_out_false, "void", linear_idx
        )

        self.builder.end_if()

        # Close all loops
        for _ in loop_vars:
            self.builder.end_for()

        return tmp_name

    def _handle_numpy_matmul_op(self, left_node, right_node):
        return self._handle_matmul_helper(left_node, right_node)

    def _handle_numpy_matmul(self, node, func_name):
        if len(node.args) != 2:
            raise NotImplementedError("matmul/dot requires 2 arguments")
        return self._handle_matmul_helper(node.args[0], node.args[1])

    def _handle_numpy_outer(self, node, func_name):
        if len(node.args) != 2:
            raise NotImplementedError("outer requires 2 arguments")

        arg0 = node.args[0]
        arg1 = node.args[1]

        if not self.la_handler:
            raise RuntimeError("LinearAlgebraHandler not initialized")

        res_a = self.la_handler.parse_arg(arg0)
        res_b = self.la_handler.parse_arg(arg1)

        # Resolve standard names if parse_arg failed (likely complex expression)
        if not res_a[0]:
            left_name = self.visit(arg0)
            arg0 = ast.Name(id=left_name)
            res_a = self.la_handler.parse_arg(arg0)

        if not res_b[0]:
            right_name = self.visit(arg1)
            arg1 = ast.Name(id=right_name)
            res_b = self.la_handler.parse_arg(arg1)

        name_a, subset_a, shape_a, indices_a = res_a
        name_b, subset_b, shape_b, indices_b = res_b

        if not name_a or not name_b:
            raise NotImplementedError("Could not resolve outer operands")

        def get_flattened_size_expr(name, indices, shapes):
            # Simplified: if slice, we use parse_arg's returned `shapes` (which are dim sizes of the slice)
            # And multiply them.
            size_expr = "1"
            for s in shapes:
                if size_expr == "1":
                    size_expr = str(s)
                else:
                    size_expr = f"({size_expr} * {str(s)})"
            return size_expr

        m_expr = get_flattened_size_expr(name_a, indices_a, shape_a)
        n_expr = get_flattened_size_expr(name_b, indices_b, shape_b)

        # Create temporary container
        # Since outer usually promotes types or uses standard types, we default to double for now.
        dtype = Scalar(PrimitiveType.Double)

        # Use helper to create array temp which handles symbol table and array info
        tmp_name = self._create_array_temp([m_expr, n_expr], dtype)

        new_call_node = ast.Call(
            func=node.func, args=[arg0, arg1], keywords=node.keywords
        )

        self.la_handler.handle_outer(tmp_name, new_call_node)

        return tmp_name

    def _handle_ufunc_outer(self, node, ufunc_name):
        """Handle np.add.outer, np.subtract.outer, np.multiply.outer, etc.

        These compute the outer operation for the given ufunc:
        - np.add.outer(a, b) -> a[:, np.newaxis] + b (outer sum)
        - np.subtract.outer(a, b) -> a[:, np.newaxis] - b (outer difference)
        - np.multiply.outer(a, b) -> a[:, np.newaxis] * b (same as np.outer)
        """
        if len(node.args) != 2:
            raise NotImplementedError(f"{ufunc_name}.outer requires 2 arguments")

        # For np.multiply.outer, use the existing GEMM-based outer handler
        if ufunc_name == "multiply":
            return self._handle_numpy_outer(node, "outer")

        # Map ufunc names to operation names and tasklet opcodes
        op_map = {
            "add": ("add", TaskletCode.fp_add, TaskletCode.int_add),
            "subtract": ("sub", TaskletCode.fp_sub, TaskletCode.int_sub),
            "divide": ("div", TaskletCode.fp_div, TaskletCode.int_sdiv),
            "minimum": ("min", CMathFunction.fmin, TaskletCode.int_smin),
            "maximum": ("max", CMathFunction.fmax, TaskletCode.int_smax),
        }

        if ufunc_name not in op_map:
            raise NotImplementedError(f"{ufunc_name}.outer not supported")

        op_name, fp_opcode, int_opcode = op_map[ufunc_name]

        # Use la_handler.parse_arg to properly handle sliced arrays
        if not self.la_handler:
            raise RuntimeError("LinearAlgebraHandler not initialized")

        arg0 = node.args[0]
        arg1 = node.args[1]

        res_a = self.la_handler.parse_arg(arg0)
        res_b = self.la_handler.parse_arg(arg1)

        # If parse_arg fails for complex expressions, try visiting and re-parsing
        if not res_a[0]:
            left_name = self.visit(arg0)
            arg0 = ast.Name(id=left_name)
            res_a = self.la_handler.parse_arg(arg0)

        if not res_b[0]:
            right_name = self.visit(arg1)
            arg1 = ast.Name(id=right_name)
            res_b = self.la_handler.parse_arg(arg1)

        name_a, subset_a, shape_a, indices_a = res_a
        name_b, subset_b, shape_b, indices_b = res_b

        if not name_a or not name_b:
            raise NotImplementedError("Could not resolve ufunc outer operands")

        # Compute flattened sizes - outer treats inputs as 1D
        def get_flattened_size_expr(shapes):
            if not shapes:
                return "1"
            size_expr = str(shapes[0])
            for s in shapes[1:]:
                size_expr = f"({size_expr} * {str(s)})"
            return size_expr

        m_expr = get_flattened_size_expr(shape_a)
        n_expr = get_flattened_size_expr(shape_b)

        # Determine output dtype - infer from inputs or default to double
        dtype_left = self._get_dtype(name_a)
        dtype_right = self._get_dtype(name_b)
        dtype = self._promote_dtypes(dtype_left, dtype_right)

        # Determine if we're working with integers
        is_int = dtype.primitive_type in [
            PrimitiveType.Int64,
            PrimitiveType.Int32,
            PrimitiveType.Int8,
            PrimitiveType.Int16,
            PrimitiveType.UInt64,
            PrimitiveType.UInt32,
            PrimitiveType.UInt8,
            PrimitiveType.UInt16,
        ]

        # Create output array with shape (M, N)
        tmp_name = self._create_array_temp([m_expr, n_expr], dtype)

        # Generate unique loop variable names
        i_var = self._get_temp_name("_outer_i_")
        j_var = self._get_temp_name("_outer_j_")

        # Ensure loop variables exist
        if not self.builder.exists(i_var):
            self.builder.add_container(i_var, Scalar(PrimitiveType.Int64), False)
            self.symbol_table[i_var] = Scalar(PrimitiveType.Int64)
        if not self.builder.exists(j_var):
            self.builder.add_container(j_var, Scalar(PrimitiveType.Int64), False)
            self.symbol_table[j_var] = Scalar(PrimitiveType.Int64)

        # Helper function to compute the linear index for a sliced array access
        def compute_linear_index(name, subset, indices, loop_var):
            """
            Compute linear index for accessing element loop_var of a sliced array.

            For array A with shape (N, M):
            - A[:, k] (column k): linear_index = loop_var * M + k
            - A[k, :] (row k): linear_index = k * M + loop_var
            - A[:] (1D array): linear_index = loop_var

            The indices list contains AST nodes showing which dims are sliced vs fixed.
            subset contains start indices for each dimension.
            """
            if not indices:
                # Simple 1D array, no slicing
                return loop_var

            info = self.array_info.get(name, {})
            shapes = info.get("shapes", [])
            ndim = info.get("ndim", len(shapes))

            if ndim == 0:
                return loop_var

            # Compute strides (row-major order)
            strides = []
            current_stride = "1"
            for i in range(ndim - 1, -1, -1):
                strides.insert(0, current_stride)
                if i > 0:
                    dim_size = shapes[i] if i < len(shapes) else f"_{name}_shape_{i}"
                    if current_stride == "1":
                        current_stride = str(dim_size)
                    else:
                        current_stride = f"({current_stride} * {dim_size})"

            # Build linear index from subset and indices info
            terms = []
            loop_var_used = False

            for i, idx in enumerate(indices):
                stride = strides[i] if i < len(strides) else "1"
                start = subset[i] if i < len(subset) else "0"

                if isinstance(idx, ast.Slice):
                    # This dimension is sliced - use loop_var
                    if stride == "1":
                        term = f"({start} + {loop_var})"
                    else:
                        term = f"(({start} + {loop_var}) * {stride})"
                    loop_var_used = True
                else:
                    # This dimension has a fixed index
                    if stride == "1":
                        term = start
                    else:
                        term = f"({start} * {stride})"

                terms.append(term)

            # Sum all terms
            if not terms:
                return loop_var

            result = terms[0]
            for t in terms[1:]:
                result = f"({result} + {t})"

            return result

        # Create nested for loops: for i in range(M): for j in range(N): C[i,j] = A[i] op B[j]
        self.builder.begin_for(i_var, "0", m_expr, "1")
        self.builder.begin_for(j_var, "0", n_expr, "1")

        # Create the assignment block: C[i, j] = A[i] op B[j]
        block = self.builder.add_block()

        # Add access nodes
        t_a = self.builder.add_access(block, name_a)
        t_b = self.builder.add_access(block, name_b)
        t_c = self.builder.add_access(block, tmp_name)

        # Determine tasklet type based on operation
        if ufunc_name in ["minimum", "maximum"]:
            # Use intrinsic for min/max
            if is_int:
                t_task = self.builder.add_tasklet(
                    block, int_opcode, ["_in1", "_in2"], ["_out"]
                )
            else:
                t_task = self.builder.add_cmath(block, fp_opcode)
        else:
            # Use regular tasklet for arithmetic ops
            tasklet_code = int_opcode if is_int else fp_opcode
            t_task = self.builder.add_tasklet(
                block, tasklet_code, ["_in1", "_in2"], ["_out"]
            )

        # Compute the linear index for A[i]
        a_index = compute_linear_index(name_a, subset_a, indices_a, i_var)

        # Compute the linear index for B[j]
        b_index = compute_linear_index(name_b, subset_b, indices_b, j_var)

        # Connect A[i + offset_a] -> tasklet
        self.builder.add_memlet(block, t_a, "void", t_task, "_in1", a_index)

        # Connect B[j + offset_b] -> tasklet
        self.builder.add_memlet(block, t_b, "void", t_task, "_in2", b_index)

        # Connect tasklet -> C[i * N + j] (linear index for 2D output)
        flat_index = f"(({i_var}) * ({n_expr}) + ({j_var}))"
        self.builder.add_memlet(block, t_task, "_out", t_c, "void", flat_index)

        self.builder.end_for()  # end j loop
        self.builder.end_for()  # end i loop

        return tmp_name

    def _op_symbol(self, op_name):
        """Convert operation name to symbol."""
        symbols = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "min": "min",  # Will need special handling
            "max": "max",  # Will need special handling
        }
        return symbols.get(op_name, op_name)

    def _handle_matmul_helper(self, left_node, right_node):
        if not self.la_handler:
            raise RuntimeError("LinearAlgebraHandler not initialized")

        res_a = self.la_handler.parse_arg(left_node)
        res_b = self.la_handler.parse_arg(right_node)

        if not res_a[0]:
            left_name = self.visit(left_node)
            left_node = ast.Name(id=left_name)
            res_a = self.la_handler.parse_arg(left_node)

        if not res_b[0]:
            right_name = self.visit(right_node)
            right_node = ast.Name(id=right_name)
            res_b = self.la_handler.parse_arg(right_node)

        name_a, subset_a, shape_a, indices_a = res_a
        name_b, subset_b, shape_b, indices_b = res_b

        if not name_a or not name_b:
            raise NotImplementedError("Could not resolve matmul operands")

        real_shape_a = shape_a
        real_shape_b = shape_b

        ndim_a = len(real_shape_a)
        ndim_b = len(real_shape_b)

        output_shape = []
        is_scalar = False

        if ndim_a == 1 and ndim_b == 1:
            is_scalar = True
            output_shape = []
        elif ndim_a == 2 and ndim_b == 2:
            output_shape = [real_shape_a[0], real_shape_b[1]]
        elif ndim_a == 2 and ndim_b == 1:
            output_shape = [real_shape_a[0]]
        elif ndim_a == 1 and ndim_b == 2:
            output_shape = [real_shape_b[1]]
        elif ndim_a > 2 or ndim_b > 2:
            if ndim_a == ndim_b:
                output_shape = list(real_shape_a[:-2]) + [
                    real_shape_a[-2],
                    real_shape_b[-1],
                ]
            else:
                raise NotImplementedError(
                    "Broadcasting with different ranks not fully supported yet"
                )
        else:
            raise NotImplementedError(
                f"Matmul with ranks {ndim_a} and {ndim_b} not supported"
            )

        dtype = Scalar(PrimitiveType.Double)

        if is_scalar:
            tmp_name = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(tmp_name, dtype, False)
            self.symbol_table[tmp_name] = dtype
        else:
            tmp_name = self._create_array_temp(output_shape, dtype)

        if ndim_a > 2 or ndim_b > 2:
            # Generate loops for broadcasting
            batch_dims = ndim_a - 2
            loop_vars = []

            for i in range(batch_dims):
                loop_var = f"_i{self._get_unique_id()}"
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                loop_vars.append(loop_var)
                dim_size = real_shape_a[i]
                self.builder.begin_for(loop_var, "0", str(dim_size), "1")

            def make_slice(name, indices):
                elts = []
                for idx in indices:
                    if idx == ":":
                        elts.append(ast.Slice())
                    else:
                        elts.append(ast.Name(id=idx))

                return ast.Subscript(
                    value=ast.Name(id=name), slice=ast.Tuple(elts=elts), ctx=ast.Load()
                )

            indices = loop_vars + [":", ":"]
            slice_a = make_slice(name_a, indices)
            slice_b = make_slice(name_b, indices)
            slice_c = make_slice(tmp_name, indices)

            self.la_handler.handle_gemm(
                slice_c, ast.BinOp(left=slice_a, op=ast.MatMult(), right=slice_b)
            )

            for _ in range(batch_dims):
                self.builder.end_for()
        else:
            if is_scalar:
                self.la_handler.handle_dot(
                    tmp_name,
                    ast.BinOp(left=left_node, op=ast.MatMult(), right=right_node),
                )
            else:
                self.la_handler.handle_gemm(
                    tmp_name,
                    ast.BinOp(left=left_node, op=ast.MatMult(), right=right_node),
                )

        return tmp_name

    def _handle_numpy_unary_op(self, node, func_name):
        args = [self.visit(arg) for arg in node.args]
        if len(args) != 1:
            raise NotImplementedError(f"Numpy function {func_name} requires 1 argument")

        op_name = func_name
        if op_name == "absolute":
            op_name = "abs"

        return self._handle_array_unary_op(op_name, args[0])

    def _handle_numpy_reduce(self, node, func_name):
        args = node.args
        keywords = {kw.arg: kw.value for kw in node.keywords}

        array_node = args[0]
        array_name = self.visit(array_node)

        if array_name not in self.array_info:
            raise ValueError(f"Reduction input must be an array, got {array_name}")

        input_shape = self.array_info[array_name]["shapes"]
        ndim = len(input_shape)

        axis = None
        if len(args) > 1:
            axis = args[1]
        elif "axis" in keywords:
            axis = keywords["axis"]

        keepdims = False
        if "keepdims" in keywords:
            keepdims_node = keywords["keepdims"]
            if isinstance(keepdims_node, ast.Constant):
                keepdims = bool(keepdims_node.value)

        axes = []
        if axis is None:
            axes = list(range(ndim))
        elif isinstance(axis, ast.Constant):  # Single axis
            val = axis.value
            if val < 0:
                val += ndim
            axes = [val]
        elif isinstance(axis, ast.Tuple):  # Multiple axes
            for elt in axis.elts:
                if isinstance(elt, ast.Constant):
                    val = elt.value
                    if val < 0:
                        val += ndim
                    axes.append(val)
        elif (
            isinstance(axis, ast.UnaryOp)
            and isinstance(axis.op, ast.USub)
            and isinstance(axis.operand, ast.Constant)
        ):
            val = -axis.operand.value
            if val < 0:
                val += ndim
            axes = [val]
        else:
            # Try to evaluate simple expression
            try:
                val = int(self.visit(axis))
                if val < 0:
                    val += ndim
                axes = [val]
            except:
                raise NotImplementedError("Dynamic axis not supported")

        # Calculate output shape
        output_shape = []
        for i in range(ndim):
            if i in axes:
                if keepdims:
                    output_shape.append("1")
            else:
                output_shape.append(input_shape[i])

        dtype = self._get_dtype(array_name)

        if not output_shape:
            tmp_name = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(tmp_name, dtype, False)
            self.symbol_table[tmp_name] = dtype
            self.array_info[tmp_name] = {"ndim": 0, "shapes": []}
        else:
            tmp_name = self._create_array_temp(output_shape, dtype)

        self.builder.add_reduce_op(
            func_name, array_name, tmp_name, input_shape, axes, keepdims
        )

        return tmp_name

    def _handle_numpy_astype(self, node, array_name):
        """Handle numpy array.astype(dtype) method calls."""
        if len(node.args) < 1:
            raise ValueError("astype requires at least one argument (dtype)")

        dtype_arg = node.args[0]
        target_dtype = self._map_numpy_dtype(dtype_arg)

        # Get input array shape
        if array_name not in self.array_info:
            raise ValueError(f"Array {array_name} not found in array_info")

        input_shape = self.array_info[array_name]["shapes"]

        # Create output array with target dtype
        tmp_name = self._create_array_temp(input_shape, target_dtype)

        # Add cast operation
        self.builder.add_cast_op(
            array_name, tmp_name, input_shape, target_dtype.primitive_type
        )

        return tmp_name

    def _handle_numpy_copy(self, node, array_name):
        """Handle numpy array.copy() method calls using memcpy."""
        if array_name not in self.array_info:
            raise ValueError(f"Array {array_name} not found in array_info")

        input_shape = self.array_info[array_name]["shapes"]

        # Get element type from array
        element_type = Scalar(PrimitiveType.Double)  # Default
        if array_name in self.symbol_table:
            sym_type = self.symbol_table[array_name]
            if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                element_type = sym_type.pointee_type

        # Create output array with same dtype
        tmp_name = self._create_array_temp(input_shape, element_type)

        # Calculate total number of bytes to copy
        # count = total_elements * sizeof(element_type)
        total_elements = " * ".join([f"({s})" for s in input_shape])
        element_size = self.builder.get_sizeof(element_type)
        count_expr = f"({total_elements}) * ({element_size})"

        # Get pointer type for memlets
        ptr_type = Pointer(element_type)

        # Add memcpy operation
        block = self.builder.add_block()
        t_src = self.builder.add_access(block, array_name)
        t_dst = self.builder.add_access(block, tmp_name)
        t_memcpy = self.builder.add_memcpy(block, count_expr)

        # Connect source and destination
        self.builder.add_memlet(block, t_src, "void", t_memcpy, "_src", "", ptr_type)
        self.builder.add_memlet(block, t_memcpy, "_dst", t_dst, "void", "", ptr_type)

        return tmp_name

    def _handle_scipy_softmax(self, node, func_name):
        args = node.args
        keywords = {kw.arg: kw.value for kw in node.keywords}

        array_node = args[0]
        array_name = self.visit(array_node)

        if array_name not in self.array_info:
            raise ValueError(f"Softmax input must be an array, got {array_name}")

        input_shape = self.array_info[array_name]["shapes"]
        ndim = len(input_shape)

        axis = None
        if len(args) > 1:
            axis = args[1]
        elif "axis" in keywords:
            axis = keywords["axis"]

        axes = []
        if axis is None:
            axes = list(range(ndim))
        elif isinstance(axis, ast.Constant):  # Single axis
            val = axis.value
            if val < 0:
                val += ndim
            axes = [val]
        elif isinstance(axis, ast.Tuple):  # Multiple axes
            for elt in axis.elts:
                if isinstance(elt, ast.Constant):
                    val = elt.value
                    if val < 0:
                        val += ndim
                    axes.append(val)
        elif (
            isinstance(axis, ast.UnaryOp)
            and isinstance(axis.op, ast.USub)
            and isinstance(axis.operand, ast.Constant)
        ):
            val = -axis.operand.value
            if val < 0:
                val += ndim
            axes = [val]
        else:
            # Try to evaluate simple expression
            try:
                val = int(self.visit(axis))
                if val < 0:
                    val += ndim
                axes = [val]
            except:
                raise NotImplementedError("Dynamic axis not supported")

        # Create output array
        # Assume double for now, or infer from input
        dtype = Scalar(PrimitiveType.Double)  # TODO: infer

        tmp_name = self._create_array_temp(input_shape, dtype)

        self.builder.add_reduce_op(
            func_name, array_name, tmp_name, input_shape, axes, False
        )

        return tmp_name
