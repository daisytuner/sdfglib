import ast
import inspect
import textwrap
from ._docc import Scalar, PrimitiveType, Pointer, Type, DebugInfo, Structure


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

    def _init_numpy_handlers(self):
        self.numpy_handlers = {
            "empty": self._handle_numpy_alloc,
            "zeros": self._handle_numpy_alloc,
            "ones": self._handle_numpy_alloc,
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
            "minimum": self._handle_numpy_binary_op,
            "maximum": self._handle_numpy_binary_op,
        }

    def generic_visit(self, node):
        return super().generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def visit_Name(self, node):
        return node.id

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

        if self.builder.has_container(expr_str):
            access = self.builder.add_access(block, expr_str, debug_info)
            try:
                self._access_cache[(block, expr_str)] = (access, "")
            except TypeError:
                pass
            return access, ""

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

            intrinsic_name = "fmax" if func_name == "max" else "fmin"
            t_task = self.builder.add_intrinsic(block, intrinsic_name)

            for i, arg in enumerate(casted_args):
                t_arg, arg_sub = self._add_read(block, arg)
                self.builder.add_memlet(
                    block, t_arg, "void", t_task, f"_in{i+1}", arg_sub
                )
        else:
            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            # Use int_smax/int_smin tasklet
            opcode = "int_smax" if func_name == "max" else "int_smin"
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
        t_task = self.builder.add_tasklet(block, "assign", ["_in"], ["_out"])
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
                    # Check if it's a method call on an array (e.g., arr.astype(...))
                    array_name = node.func.value.id
                    method_name = node.func.attr
                    if array_name in self.array_info and method_name == "astype":
                        return self._handle_numpy_astype(node, array_name)
            elif isinstance(node.func.value, ast.Attribute):
                if (
                    isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "scipy"
                    and node.func.value.attr == "special"
                ):
                    if node.func.attr == "softmax":
                        return self._handle_scipy_softmax(node, "softmax")

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

        math_funcs = [
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "pow",
            "abs",
            "ceil",
            "floor",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
        ]

        if func_name in math_funcs:
            args = [self.visit(arg) for arg in node.args]

            tmp_name = self._get_temp_name("_tmp_")
            dtype = Scalar(PrimitiveType.Double)
            self.builder.add_container(tmp_name, dtype, False)
            self.symbol_table[tmp_name] = dtype

            block = self.builder.add_block()
            t_out = self.builder.add_access(block, tmp_name)

            t_task = self.builder.add_intrinsic(block, func_name)

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
            def __init__(self, suffix, globals_dict):
                self.suffix = suffix
                self.globals_dict = globals_dict

            def visit_Name(self, node):
                if node.id in self.globals_dict:
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
        right = self.visit(node.right)
        op = self.visit(node.op)

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
                t_task = self.builder.add_tasklet(c_block, "assign", ["_in"], ["_out"])
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
                t_task = self.builder.add_tasklet(c_block, "assign", ["_in"], ["_out"])
                self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

                real_right = right_cast

        # Special cases
        if op == "**":
            block = self.builder.add_block()
            t_left, left_sub = self._add_read(block, real_left)
            t_right, right_sub = self._add_read(block, real_right)
            t_out = self.builder.add_access(block, tmp_name)

            t_task = self.builder.add_intrinsic(block, "pow")
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
                    block, "int_rem", ["_in1", "_in2"], ["_out"]
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
                    block, "int_add", ["_in1", "_in2"], ["_out"]
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
                    block, "int_rem", ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_add_out, "void", t_rem2, "_in1", "")
                self.builder.add_memlet(
                    block, t_right, "void", t_rem2, "_in2", right_sub
                )
                self.builder.add_memlet(block, t_rem2, "_out", t_out, "void", "")

                return tmp_name
            else:
                t_task = self.builder.add_intrinsic(block, "fmod")
                self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
                self.builder.add_memlet(
                    block, t_right, "void", t_task, "_in2", right_sub
                )
                self.builder.add_memlet(block, t_task, "_out", t_out, "void", "")
                return tmp_name

        prefix = "int" if dtype.primitive_type == PrimitiveType.Int64 else "fp"
        op_name = ""
        if op == "+":
            op_name = "add"
        elif op == "-":
            op_name = "sub"
        elif op == "*":
            op_name = "mul"
        elif op == "/":
            op_name = "div"
        elif op == "//":
            op_name = "div"
        elif op == "|":
            op_name = "or"
        elif op == "^":
            op_name = "xor"

        block = self.builder.add_block()
        t_left, left_sub = self._add_read(block, real_left)
        t_right, right_sub = self._add_read(block, real_right)
        t_out = self.builder.add_access(block, tmp_name)

        tasklet_code = f"{prefix}_{op_name}"
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
        t_task = self.builder.add_tasklet(block, "assign", ["_in"], ["_out"])
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
        expr_str = f"{left} {op} {right}"

        tmp_name = f"_tmp_{self._get_unique_id()}"
        dtype = Scalar(PrimitiveType.Bool)
        self.builder.add_container(tmp_name, dtype, False)

        # Use control flow to assign boolean value
        self.builder.begin_if(expr_str)
        self.builder.add_assignment(tmp_name, "true")
        self.builder.begin_else()
        self.builder.add_assignment(tmp_name, "false")
        self.builder.end_if()

        self.symbol_table[tmp_name] = dtype
        return tmp_name

    def visit_UnaryOp(self, node):
        op = self.visit(node.op)
        operand = self.visit(node.operand)

        tmp_name = f"_tmp_{self._get_unique_id()}"
        dtype = Scalar(PrimitiveType.Double)
        if operand in self.symbol_table:
            dtype = self.symbol_table[operand]
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
                block, "int_xor", ["_in1", "_in2"], ["_out"]
            )
            self.builder.add_memlet(block, t_src, "void", t_task, "_in1", src_sub)
            self.builder.add_memlet(block, t_const, "void", t_task, "_in2", "")
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

        elif op == "-":
            if dtype.primitive_type == PrimitiveType.Int64:
                t_const = self.builder.add_constant(block, "0", dtype)
                t_task = self.builder.add_tasklet(
                    block, "int_sub", ["_in1", "_in2"], ["_out"]
                )
                self.builder.add_memlet(block, t_const, "void", t_task, "_in1", "")
                self.builder.add_memlet(block, t_src, "void", t_task, "_in2", src_sub)
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
            else:
                t_task = self.builder.add_tasklet(block, "fp_neg", ["_in"], ["_out"])
                self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
                self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")
        else:
            t_task = self.builder.add_tasklet(block, "assign", ["_in"], ["_out"])
            self.builder.add_memlet(block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "")

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
                            block, "assign", ["_in"], ["_out"]
                        )

                        # Use member index in the subset to select the correct member
                        subset = "0," + str(member_index)
                        self.builder.add_memlet(
                            block, obj_access, "", tasklet, "_in", subset
                        )
                        self.builder.add_memlet(block, tasklet, "_out", tmp_access, "")

                        return tmp_name

        raise NotImplementedError(f"Attribute access {node.attr} not supported")

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

            for idx in indices_nodes:
                if isinstance(idx, ast.Slice):
                    raise ValueError("Slices not supported in expression indexing")

            if isinstance(node.slice, ast.Tuple):
                indices = [self.visit(elt) for elt in node.slice.elts]
            else:
                indices = [self.visit(node.slice)]

            if len(indices) != ndim:
                raise ValueError(
                    f"Array {value_str} has {ndim} dimensions, but accessed with {len(indices)} indices"
                )

            linear_index = ""
            for i in range(ndim):
                term = indices[i]
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
                t_task = self.builder.add_tasklet(block, "assign", ["_in"], ["_out"])

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

    def _create_array_temp(self, shape, dtype, zero_init=False, ones_init=False):
        tmp_name = f"_tmp_{self._get_unique_id()}"

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
        self.array_info[tmp_name] = {"ndim": len(shape), "shapes": shape}

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
            if not self.builder.has_container(loop_var):
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

            t_task = self.builder.add_tasklet(block_assign, "assign", ["_in"], ["_out"])
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

        tmp_name = self._create_array_temp(shape, dtype)

        # Add operation
        self.builder.add_elementwise_unary_op(op_type, operand, tmp_name, shape)

        return tmp_name

    def _handle_array_binary_op(self, op_type, left, right):
        # Determine output shape
        shape = []
        if left in self.array_info:
            shape = self.array_info[left]["shapes"]
        elif right in self.array_info:
            shape = self.array_info[right]["shapes"]

        # Determine dtype
        dtype_left = self._get_dtype(left)
        dtype_right = self._get_dtype(right)

        assert dtype_left.primitive_type == dtype_right.primitive_type
        dtype = dtype_left

        tmp_name = self._create_array_temp(shape, dtype)

        # Add operation
        self.builder.add_elementwise_op(op_type, left, right, tmp_name, shape)

        return tmp_name

    def _handle_numpy_alloc(self, node, func_name):
        # Parse shape
        shape_arg = node.args[0]
        dims = []
        if isinstance(shape_arg, ast.Tuple):
            dims = [self.visit(elt) for elt in shape_arg.elts]
        elif isinstance(shape_arg, ast.List):
            dims = [self.visit(elt) for elt in shape_arg.elts]
        else:
            val = self.visit(shape_arg)
            if val.startswith("_shape_proxy_"):
                array_name = val[len("_shape_proxy_") :]
                if array_name in self.array_info:
                    dims = self.array_info[array_name]["shapes"]
                else:
                    dims = [val]
            else:
                dims = [val]

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
        if not self.builder.has_container(loop_var):
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

        t_task = self.builder.add_tasklet(block_assign, "assign", ["_in"], ["_out"])
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

    def _handle_numpy_matmul_op(self, left_node, right_node):
        return self._handle_matmul_helper(left_node, right_node)

    def _handle_numpy_matmul(self, node, func_name):
        if len(node.args) != 2:
            raise NotImplementedError("matmul/dot requires 2 arguments")
        return self._handle_matmul_helper(node.args[0], node.args[1])

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
            self.la_handler.handle_gemm(
                tmp_name, ast.BinOp(left=left_node, op=ast.MatMult(), right=right_node)
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
