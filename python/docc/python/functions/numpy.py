import ast
from docc.sdfg import (
    Scalar,
    PrimitiveType,
    Pointer,
    DebugInfo,
    TaskletCode,
    CMathFunction,
    Tensor,
)
from docc.python.types import (
    element_type_from_ast_node,
    promote_element_types,
)
from docc.python.ast_utils import get_debug_info


class NumPyHandler:
    """
    Unified handler for NumPy operations including:
    - Array creation (empty, zeros, ones, eye, etc.)
    - Elementwise operations (add, subtract, multiply, etc.)
    - Linear algebra (matmul, dot, outer, gemm)
    - Array manipulation (transpose)
    - Reductions (sum, max, min, mean, std)
    """

    def __init__(self, expression_visitor):
        self._ev = expression_visitor
        self._unique_counter = 0
        self.function_handlers = {
            "empty": self._handle_numpy_alloc,
            "empty_like": self._handle_numpy_empty_like,
            "zeros": self._handle_numpy_alloc,
            "zeros_like": self._handle_numpy_zeros_like,
            "ones": self._handle_numpy_alloc,
            "ndarray": self._handle_numpy_alloc,
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
            "clip": self._handle_numpy_clip,
            "transpose": self._handle_numpy_transpose,
        }

    # Expose parent properties for convenience
    @property
    def tensor_table(self):
        return self._ev.tensor_table

    @property
    def builder(self):
        return self._ev.builder

    @property
    def container_table(self):
        return self._ev.container_table

    @property
    def globals_dict(self):
        return self._ev.globals_dict

    @property
    def shapes_runtime_info(self):
        return self._ev.shapes_runtime_info

    def _get_unique_id(self):
        return self._ev._get_unique_id()

    def _add_read(self, block, expr_str, debug_info=None):
        return self._ev._add_read(block, expr_str, debug_info)

    def _is_int(self, operand):
        return self._ev._is_int(operand)

    def visit(self, node):
        return self._ev.visit(node)

    # ========== Linear Algebra Helper Methods (from LinearAlgebraHandler) ==========

    def parse_arg(self, node):
        """Parse an array argument, returning (name, start_indices, slice_shape, indices)."""
        if isinstance(node, ast.Name):
            if node.id in self.tensor_table:
                return node.id, [], self.tensor_table[node.id].shape, []
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.tensor_table:
                name = node.value.id
                indices = []
                if isinstance(node.slice, ast.Tuple):
                    indices = node.slice.elts
                else:
                    indices = [node.slice]

                start_indices = []
                slice_shape = []

                for i, idx in enumerate(indices):
                    if isinstance(idx, ast.Slice):
                        start = "0"
                        if idx.lower:
                            start = self._ev.visit(idx.lower)
                        start_indices.append(start)

                        shapes = self.tensor_table[name].shape
                        dim_size = (
                            shapes[i] if i < len(shapes) else f"_{name}_shape_{i}"
                        )
                        stop = dim_size
                        if idx.upper:
                            stop = self._ev.visit(idx.upper)

                        size = f"({stop} - {start})"
                        slice_shape.append(size)
                    else:
                        if isinstance(idx, ast.Name) and idx.id in self.tensor_table:
                            # This is an array index (gather operation)
                            return None, None, None, None
                        val = self._ev.visit(idx)
                        start_indices.append(val)

                return name, start_indices, slice_shape, indices

        return None, None, None, None

    def flatten_subset(self, name, start_indices):
        """Convert multi-dimensional start indices to a flattened linear offset."""
        if not start_indices:
            return []
        info = self.tensor_table[name]
        shapes = info.shape
        ndim = len(info.shape)

        if len(start_indices) != ndim:
            return start_indices

        strides = []
        current_stride = "1"
        strides.append(current_stride)
        for i in range(ndim - 1, 0, -1):
            dim_size = shapes[i]
            if current_stride == "1":
                current_stride = str(dim_size)
            else:
                current_stride = f"({current_stride} * {dim_size})"
            strides.append(current_stride)
        strides = list(reversed(strides))

        offset = "0"
        for i in range(ndim):
            idx = start_indices[i]
            stride = strides[i]
            term = f"({idx} * {stride})" if stride != "1" else idx
            if offset == "0":
                offset = term
            else:
                offset = f"({offset} + {term})"

        return [offset]

    def is_gemm(self, node):
        """Check if a node represents a GEMM operation (matrix multiplication)."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "dot":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "matmul":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "matmul":
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.is_gemm(node.left) or self.is_gemm(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self.is_gemm(node.left) or self.is_gemm(node.right)
        return False

    def _is_stride_1(self, name, indices):
        """Check if the sliced dimension has stride 1 (contiguous access)."""
        if name not in self.tensor_table:
            return True
        info = self.tensor_table[name]
        ndim = len(info.shape)

        if not indices:
            return True

        sliced_dim = -1
        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                sliced_dim = i
                break

        if sliced_dim == -1:
            if len(indices) < ndim:
                sliced_dim = ndim - 1
            else:
                return True

        return sliced_dim == ndim - 1

    def _is_target(self, node, target_name):
        """Check if node refers to the target."""
        if isinstance(target_name, ast.AST):
            return self._ev.visit(node) == self._ev.visit(target_name)

        if isinstance(node, ast.Name) and node.id == target_name:
            return True
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == target_name:
                return True
        return False

    def _is_dot_call(self, node):
        """Check if node is a dot product call."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "dot":
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        return False

    def handle_gemm(self, target, value_node):
        """Handle GEMM (General Matrix Multiply) operations: C = alpha * A @ B + beta * C."""
        target_name = None
        target_subset = []

        if isinstance(target, str):
            target_name = target
        elif isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name):
                res = self.parse_arg(target)
                if res[0]:
                    target_name = res[0]
                    target_subset = self.flatten_subset(target_name, res[1])
                else:
                    target_name = target.value.id

        if not target_name or target_name not in self.tensor_table:
            return False

        alpha = "1.0"
        beta = "0.0"
        A = None
        B = None

        def extract_factor(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if self.is_gemm(node.left):
                    return node.left, self._ev.visit(node.right)
                if self.is_gemm(node.right):
                    return node.right, self._ev.visit(node.left)

                res = self.parse_arg(node.left)
                if res[0]:
                    return node.left, self._ev.visit(node.right)
                res = self.parse_arg(node.right)
                if res[0]:
                    return node.right, self._ev.visit(node.left)
            return node, "1.0"

        def parse_term(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                l, l_f = extract_factor(node.left)
                r, r_f = extract_factor(node.right)
                f = "1.0"
                if l_f != "1.0":
                    f = l_f
                if r_f != "1.0":
                    if f == "1.0":
                        f = r_f
                    else:
                        f = f"({f} * {r_f})"
                return l, r, f

            if isinstance(node, ast.Call):
                is_gemm_call = False
                if isinstance(node.func, ast.Attribute) and node.func.attr in [
                    "dot",
                    "matmul",
                ]:
                    is_gemm_call = True
                if isinstance(node.func, ast.Name) and node.func.id in [
                    "dot",
                    "matmul",
                ]:
                    is_gemm_call = True

                if is_gemm_call and len(node.args) == 2:
                    return node.args[0], node.args[1], "1.0"

            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                l, r, a = parse_term(node.left)
                if l:
                    return l, r, self._ev.visit(node.right)
                l, r, a = parse_term(node.right)
                if l:
                    return l, r, self._ev.visit(node.left)

            return None, None, None

        if isinstance(value_node, ast.BinOp) and isinstance(value_node.op, ast.Add):
            l, r, a = parse_term(value_node.left)
            if l:
                A = l
                B = r
                alpha = a
                if isinstance(value_node.right, ast.BinOp) and isinstance(
                    value_node.right.op, ast.Mult
                ):
                    if self._is_target(value_node.right.left, target_name):
                        beta = self._ev.visit(value_node.right.right)
                    elif self._is_target(value_node.right.right, target_name):
                        beta = self._ev.visit(value_node.right.left)
                elif self._is_target(value_node.right, target_name):
                    beta = "1.0"
            else:
                l, r, a = parse_term(value_node.right)
                if l:
                    A = l
                    B = r
                    alpha = a
                    if isinstance(value_node.left, ast.BinOp) and isinstance(
                        value_node.left.op, ast.Mult
                    ):
                        if self._is_target(value_node.left.left, target_name):
                            beta = self._ev.visit(value_node.left.right)
                        elif self._is_target(value_node.left.right, target_name):
                            beta = self._ev.visit(value_node.left.left)
                    elif self._is_target(value_node.left, target_name):
                        beta = "1.0"
        else:
            l, r, a = parse_term(value_node)
            if l:
                A = l
                B = r
                alpha = a

        if A is None or B is None:
            return False

        def get_name_and_trans(node):
            if isinstance(node, ast.Attribute) and node.attr == "T":
                return node.value, True
            return node, False

        A_node, trans_a = get_name_and_trans(A)
        B_node, trans_b = get_name_and_trans(B)

        if self.is_gemm(A_node):
            tmp_name = self._ev.visit(A_node)
            A_node = ast.Name(id=tmp_name)

        if self.is_gemm(B_node):
            tmp_name = self._ev.visit(B_node)
            B_node = ast.Name(id=tmp_name)

        res_a = self.parse_arg(A_node)
        res_b = self.parse_arg(B_node)

        if not res_a[0] or not res_b[0]:
            return False

        A_name, subset_a, shape_a, indices_a = res_a
        B_name, subset_b, shape_b, indices_b = res_b

        flat_subset_a = self.flatten_subset(A_name, subset_a)
        flat_subset_b = self.flatten_subset(B_name, subset_b)

        def get_ndim(name):
            if name not in self.tensor_table:
                return 1
            return len(self.tensor_table[name].shape)

        if len(shape_a) == 2:
            if not trans_a:
                m = shape_a[0]
                k = shape_a[1]
            else:
                m = shape_a[1]
                k = shape_a[0]
        else:
            m = "1"
            k = shape_a[0]
            if self._is_stride_1(A_name, indices_a):
                if get_ndim(A_name) == 1:
                    trans_a = True
                else:
                    trans_a = False
            else:
                trans_a = True

        if len(shape_b) == 2:
            if not trans_b:
                n = shape_b[1]
            else:
                n = shape_b[0]
        else:
            n = "1"
            if self._is_stride_1(B_name, indices_b):
                if get_ndim(B_name) == 1:
                    trans_b = False
                else:
                    trans_b = True
            else:
                trans_b = False

        def get_ld(name):
            if name not in self.tensor_table:
                return ""
            shapes = self.tensor_table[name].shape
            if len(shapes) >= 2:
                return str(shapes[1])
            return "1"

        lda = get_ld(A_name)
        ldb = get_ld(B_name)

        ldc = ""
        if target_name:
            if get_ndim(target_name) == 1 and m == "1":
                ldc = n
            else:
                ldc = get_ld(target_name)

        self.builder.add_gemm(
            A_name,
            B_name,
            target_name,
            alpha,
            beta,
            m,
            n,
            k,
            trans_a,
            trans_b,
            flat_subset_a,
            flat_subset_b,
            target_subset,
            lda,
            ldb,
            ldc,
        )
        return True

    def handle_dot(self, target, value_node):
        """Handle dot product operations for 1D vectors."""
        dot_node = None
        is_accumulate = False

        if self._is_dot_call(value_node):
            dot_node = value_node
        elif isinstance(value_node, ast.BinOp) and isinstance(value_node.op, ast.Add):
            if self._is_dot_call(value_node.left):
                dot_node = value_node.left
                if self._is_target(value_node.right, target):
                    is_accumulate = True
            elif self._is_dot_call(value_node.right):
                dot_node = value_node.right
                if self._is_target(value_node.left, target):
                    is_accumulate = True

        if not dot_node:
            return False

        arg0 = None
        arg1 = None

        if isinstance(dot_node, ast.Call):
            args = dot_node.args
            if len(args) != 2:
                return False
            arg0 = args[0]
            arg1 = args[1]
        elif isinstance(dot_node, ast.BinOp) and isinstance(dot_node.op, ast.MatMult):
            arg0 = dot_node.left
            arg1 = dot_node.right

        res_a = self.parse_arg(arg0)
        res_b = self.parse_arg(arg1)

        if not res_a[0] or not res_b[0]:
            return False

        name_a, subset_a, shape_a, indices_a = res_a
        name_b, subset_b, shape_b, indices_b = res_b

        if len(shape_a) != 1 or len(shape_b) != 1:
            return False

        n = shape_a[0]

        def get_stride(name, indices):
            if not indices:
                return "1"
            info = self.tensor_table[name]
            shapes = info.shape
            ndim = len(info.shape)

            sliced_dim = -1
            for i, idx in enumerate(indices):
                if isinstance(idx, ast.Slice):
                    sliced_dim = i
                    break

            if sliced_dim == -1:
                return "1"

            stride = "1"
            for i in range(sliced_dim + 1, ndim):
                dim_size = shapes[i] if i < len(shapes) else f"_{name}_shape_{i}"
                if stride == "1":
                    stride = str(dim_size)
                else:
                    stride = f"({stride} * {dim_size})"
            return stride

        incx = get_stride(name_a, indices_a)
        incy = get_stride(name_b, indices_b)

        flat_subset_a = self.flatten_subset(name_a, subset_a)
        flat_subset_b = self.flatten_subset(name_b, subset_b)

        tmp_res = f"_dot_res_{self._get_unique_id()}"
        self.builder.add_container(tmp_res, Scalar(PrimitiveType.Double), False)
        block = self.builder.add_block()
        constant = self.builder.add_constant(block, "0.0", Scalar(PrimitiveType.Double))
        tasklet = self.builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
        self.builder.add_memlet(
            block, constant, "", tasklet, "_in", "", Scalar(PrimitiveType.Double)
        )
        access = self.builder.add_access(block, tmp_res)
        self.builder.add_memlet(
            block, tasklet, "_out", access, "", "", Scalar(PrimitiveType.Double)
        )

        self.container_table[tmp_res] = Scalar(PrimitiveType.Double)

        self.builder.add_dot(
            name_a, name_b, tmp_res, n, incx, incy, flat_subset_a, flat_subset_b
        )

        target_str = target if isinstance(target, str) else self._ev.visit(target)

        if not self.builder.exists(target_str):
            self.builder.add_container(target_str, Scalar(PrimitiveType.Double), False)
            self.container_table[target_str] = Scalar(PrimitiveType.Double)

        if is_accumulate:
            self.builder.add_assignment(target_str, f"{target_str} + {tmp_res}")
        else:
            self.builder.add_assignment(target_str, tmp_res)

        return True

    def is_outer(self, node):
        """Check if a node represents an outer product operation."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "outer":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "outer":
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.is_outer(node.left) or self.is_outer(node.right)
        return False

    def handle_outer(self, target, value_node):
        """Handle outer product operations."""
        target_name = None
        target_subset = []

        if isinstance(target, str):
            target_name = target
        elif isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, ast.Subscript):
            res = self.parse_arg(target)
            if res[0]:
                target_name = res[0]
                target_subset = self.flatten_subset(target_name, res[1])
            else:
                if isinstance(target.value, ast.Name):
                    target_name = target.value.id

        if not target_name:
            return False

        outer_calls = []
        target_found = False
        terms = []

        def collect_terms(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                collect_terms(node.left)
                collect_terms(node.right)
            else:
                terms.append(node)

        collect_terms(value_node)

        for term in terms:
            if self._is_target(term, target_name):
                target_found = True
            elif isinstance(term, ast.Call) and (
                (isinstance(term.func, ast.Attribute) and term.func.attr == "outer")
                or (isinstance(term.func, ast.Name) and term.func.id == "outer")
            ):
                if len(term.args) != 2:
                    return False
                outer_calls.append(term)
            else:
                return False

        if not outer_calls:
            return False

        parsed_outers = []
        for outer_node in outer_calls:
            arg0 = outer_node.args[0]
            arg1 = outer_node.args[1]

            res_a = self.parse_arg(arg0)
            res_b = self.parse_arg(arg1)

            if not res_a[0] or not res_b[0]:
                return False

            parsed_outers.append((res_a, res_b))

        alpha = "1.0"
        beta = "1.0" if target_found else "0.0"

        def get_flattened_size(name, indices, shapes):
            size_expr = "1"
            for s in shapes:
                if size_expr == "1":
                    size_expr = str(s)
                else:
                    size_expr = f"({size_expr} * {str(s)})"
            return size_expr

        def get_ld_2d(name):
            if name in self.tensor_table:
                shapes = self.tensor_table[name].shape
                if len(shapes) >= 2:
                    return str(shapes[1])
            return "1"

        ldc = get_ld_2d(target_name)

        for res_a, res_b in parsed_outers:
            name_a, subset_a, shape_a, indices_a = res_a
            name_b, subset_b, shape_b, indices_b = res_b

            m = get_flattened_size(name_a, indices_a, shape_a)
            n = get_flattened_size(name_b, indices_b, shape_b)
            k = "1"

            trans_a = False
            trans_b = True

            flat_subset_a = self.flatten_subset(name_a, subset_a)
            flat_subset_b = self.flatten_subset(name_b, subset_b)

            lda = "1"
            ldb = "1"

            self.builder.add_gemm(
                name_a,
                name_b,
                target_name,
                alpha,
                beta,
                m,
                n,
                k,
                trans_a,
                trans_b,
                flat_subset_a,
                flat_subset_b,
                target_subset,
                lda,
                ldb,
                ldc,
            )
            beta = "1.0"

        return True

    # ========== Transpose Operations ==========

    def _parse_perm(self, node):
        """Parse a permutation list or tuple from an AST node."""
        if isinstance(node, (ast.List, ast.Tuple)):
            res = []
            for elt in node.elts:
                val = self._ev.visit(elt)
                res.append(int(val))
            return res
        return []

    def is_transpose(self, node):
        """Check if a node represents a transpose operation."""
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
        """Handle transpose operations including .T and np.transpose()."""
        if not self.is_transpose(value_node):
            return False

        input_node = None
        perm = []

        if isinstance(value_node, ast.Attribute) and value_node.attr == "T":
            input_node = value_node.value
            perm = []  # Empty means reverse

        elif isinstance(value_node, ast.Call):
            args = value_node.args
            keywords = value_node.keywords

            is_numpy_func = False
            if isinstance(value_node.func, ast.Attribute):
                caller = ""
                if isinstance(value_node.func.value, ast.Name):
                    caller = value_node.func.value.id
                if caller in ["np", "numpy"]:
                    is_numpy_func = True
            elif isinstance(value_node.func, ast.Name):
                is_numpy_func = True

            if is_numpy_func:
                if len(args) < 1:
                    return False
                input_node = args[0]
                if len(args) > 1:
                    perm = self._parse_perm(args[1])
                for kw in keywords:
                    if kw.arg == "axes":
                        perm = self._parse_perm(kw.value)
            else:
                if isinstance(value_node.func, ast.Attribute):
                    input_node = value_node.func.value
                else:
                    return False
                if len(args) > 0:
                    perm = self._parse_perm(args[0])
                for kw in keywords:
                    if kw.arg == "axes":
                        perm = self._parse_perm(kw.value)

        input_name = self._ev.visit(input_node)
        if input_name not in self.tensor_table:
            return False

        in_info = self.tensor_table[input_name]
        in_shape = in_info.shape
        in_strings = [str(s) for s in in_shape]

        if not perm:
            perm = list(range(len(in_shape)))[::-1]

        out_shape = [in_strings[p] for p in perm]

        target_name = ""
        if isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, str):
            target_name = target

        dtype = Scalar(PrimitiveType.Double)
        if input_name in self.container_table:
            input_type = self.container_table[input_name]
            if isinstance(input_type, Pointer):
                dtype = input_type.pointee_type
            else:
                dtype = input_type

        ptr_type = Pointer(dtype)

        if not self.builder.exists(target_name):
            self.builder.add_container(target_name, ptr_type, False)
            self.container_table[target_name] = ptr_type
            self.tensor_table[target_name] = Tensor(dtype, out_shape)

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

    def handle_transpose_expr(self, node):
        """Handle .T attribute access in expressions, returning a temp array name."""
        if not isinstance(node, ast.Attribute) or node.attr != "T":
            return None

        input_name = self._ev.visit(node.value)
        if input_name not in self.tensor_table:
            return None

        in_info = self.tensor_table[input_name]
        in_shape = in_info.shape
        in_strings = [str(s) for s in in_shape]
        perm = list(range(len(in_shape)))[::-1]
        out_shape = [in_strings[p] for p in perm]

        dtype = Scalar(PrimitiveType.Double)
        if input_name in self.container_table:
            input_type = self.container_table[input_name]
            if isinstance(input_type, Pointer):
                dtype = input_type.pointee_type
            else:
                dtype = input_type

        tmp_name = self._create_array_temp(out_shape, dtype)

        debug_info = get_debug_info(node, getattr(self.builder, "filename", ""), "")
        self.builder.add_transpose(input_name, tmp_name, in_strings, perm, debug_info)

        return tmp_name

    def _handle_numpy_transpose(self, node, func_name):
        """Handle np.transpose(arr, axes=...) function call."""
        if len(node.args) < 1:
            raise ValueError("np.transpose requires at least one argument")

        input_node = node.args[0]
        input_name = self.visit(input_node)

        if input_name not in self.tensor_table:
            raise ValueError(f"Array {input_name} not found in tensor_table")

        in_info = self.tensor_table[input_name]
        in_shape = in_info.shape
        in_strings = [str(s) for s in in_shape]

        perm = []
        if len(node.args) > 1:
            perm = self._parse_perm(node.args[1])
        for kw in node.keywords:
            if kw.arg == "axes":
                perm = self._parse_perm(kw.value)

        if not perm:
            perm = list(range(len(in_shape)))[::-1]

        out_shape = [in_strings[p] for p in perm]

        dtype = Scalar(PrimitiveType.Double)
        if input_name in self.container_table:
            input_type = self.container_table[input_name]
            if isinstance(input_type, Pointer):
                dtype = input_type.pointee_type
            else:
                dtype = input_type

        tmp_name = self._create_array_temp(out_shape, dtype)

        debug_info = get_debug_info(node, getattr(self.builder, "filename", ""), "")
        self.builder.add_transpose(input_name, tmp_name, in_strings, perm, debug_info)

        return tmp_name

    def handle_numpy_call(self, node, func_name):
        if func_name in self.function_handlers:
            return self.function_handlers[func_name](node, func_name)
        raise NotImplementedError(f"NumPy function {func_name} not supported")

    def has_handler(self, func_name):
        return func_name in self.function_handlers

    def handle_array_unary_op(self, op_type, operand):
        shape = []
        if operand in self.tensor_table:
            shape = self.tensor_table[operand].shape

        dtype = self._ev._element_type(operand)

        if not shape or len(shape) == 0:
            tmp_name = self._create_array_temp(shape, dtype)

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

            self.builder.add_memlet(block, t_src, "void", t_task, "_in1", "", dtype)
            self.builder.add_memlet(block, t_task, "_out", t_dst, "void", "", dtype)

            return tmp_name

        tmp_name = self._create_array_temp(shape, dtype)
        self.builder.add_elementwise_unary_op(op_type, operand, tmp_name, shape)

        return tmp_name

    def handle_array_binary_op(self, op_type, left, right):
        left_shape = []
        right_shape = []
        if left in self.tensor_table:
            left_shape = self.tensor_table[left].shape
        if right in self.tensor_table:
            right_shape = self.tensor_table[right].shape

        shape = self._compute_broadcast_shape(left_shape, right_shape)

        dtype_left = self._ev._element_type(left)
        dtype_right = self._ev._element_type(right)
        dtype = promote_element_types(dtype_left, dtype_right)

        real_left = left
        real_right = right

        left_is_scalar = left not in self.tensor_table
        right_is_scalar = right not in self.tensor_table

        # Cast left operand if needed
        if left_is_scalar and dtype_left.primitive_type != dtype.primitive_type:
            left_cast = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(left_cast, dtype, False)
            self.container_table[left_cast] = dtype

            c_block = self.builder.add_block()
            t_src, src_sub = self._add_read(c_block, left)
            t_dst = self.builder.add_access(c_block, left_cast)
            t_task = self.builder.add_tasklet(
                c_block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

            real_left = left_cast

        # Cast right operand if needed
        if right_is_scalar and dtype_right.primitive_type != dtype.primitive_type:
            right_cast = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(right_cast, dtype, False)
            self.container_table[right_cast] = dtype

            c_block = self.builder.add_block()
            t_src, src_sub = self._add_read(c_block, right)
            t_dst = self.builder.add_access(c_block, right_cast)
            t_task = self.builder.add_tasklet(
                c_block, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(c_block, t_src, "void", t_task, "_in", src_sub)
            self.builder.add_memlet(c_block, t_task, "_out", t_dst, "void", "")

            real_right = right_cast

        # Broadcast arrays if needed
        if not left_is_scalar and self._needs_broadcast(left_shape, shape):
            real_left = self._broadcast_array(real_left, left_shape, shape, dtype)

        if not right_is_scalar and self._needs_broadcast(right_shape, shape):
            real_right = self._broadcast_array(real_right, right_shape, shape, dtype)

        tmp_name = self._create_array_temp(shape, dtype)
        self.builder.add_elementwise_op(op_type, real_left, real_right, tmp_name, shape)

        return tmp_name

    def handle_array_negate(self, operand):
        shape = self.tensor_table[operand].shape
        dtype = self._ev._element_type(operand)

        tmp_name = self._create_array_temp(shape, dtype)

        zero_name = f"_tmp_{self._get_unique_id()}"
        self.builder.add_container(zero_name, dtype, False)
        self.container_table[zero_name] = dtype

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

        self.builder.add_elementwise_op("sub", zero_name, operand, tmp_name, shape)

        return tmp_name

    def handle_array_compare(self, left, op, right, left_is_array, right_is_array):
        """Handle elementwise comparison of arrays, returning a boolean array."""
        if left_is_array:
            shape = self.tensor_table[left].shape
            arr_name = left
        else:
            shape = self.tensor_table[right].shape
            arr_name = right

        use_int_cmp = False
        arr_dtype = self._ev._element_type(arr_name)
        if arr_dtype.primitive_type in (PrimitiveType.Int32, PrimitiveType.Int64):
            use_int_cmp = True

        dtype = Scalar(PrimitiveType.Bool)
        tmp_name = self._create_array_temp(shape, dtype)

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

        scalar_name = None
        if not left_is_array:
            scalar_name = left
        elif not right_is_array:
            scalar_name = right

        if scalar_name is not None and not use_int_cmp:
            if self._is_int(scalar_name):
                float_name = f"_tmp_{self._get_unique_id()}"
                self.builder.add_container(
                    float_name, Scalar(PrimitiveType.Double), False
                )
                self.container_table[float_name] = Scalar(PrimitiveType.Double)

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

                if not left_is_array:
                    left = float_name
                else:
                    right = float_name

        loop_vars = []
        for i, dim in enumerate(shape):
            loop_var = f"_cmp_i{i}_{self._get_unique_id()}"
            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)
            loop_vars.append(loop_var)
            self.builder.begin_for(loop_var, "0", str(dim), "1")

        linear_idx = self._compute_linear_index(loop_vars, shape, tmp_name, len(shape))

        block = self.builder.add_block()

        if left_is_array:
            t_left = self.builder.add_access(block, left)
            left_sub = linear_idx
        else:
            t_left, left_sub = self._add_read(block, left)

        if right_is_array:
            t_right = self.builder.add_access(block, right)
            right_sub = linear_idx
        else:
            t_right, right_sub = self._add_read(block, right)

        t_out = self.builder.add_access(block, tmp_name)

        t_task = self.builder.add_tasklet(
            block, tasklet_code, ["_in1", "_in2"], ["_out"]
        )

        self.builder.add_memlet(block, t_left, "void", t_task, "_in1", left_sub)
        self.builder.add_memlet(block, t_right, "void", t_task, "_in2", right_sub)
        self.builder.add_memlet(block, t_task, "_out", t_out, "void", linear_idx)

        for _ in loop_vars:
            self.builder.end_for()

        return tmp_name

    # ========== NumPy Function Handlers ==========

    def _handle_numpy_alloc(self, node, func_name):
        """Handle np.empty, np.zeros, np.ones, np.ndarray."""
        shape_arg = node.args[0]
        dims = []
        dims_runtime = []
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
                if array_name in self.tensor_table:
                    info = self.tensor_table[array_name]
                    dims = info.shape
                    dims_runtime = self.shapes_runtime_info.get(array_name, dims)
                else:
                    dims = [val]
                    dims_runtime = [runtime_val]
            else:
                dims = [val]
                dims_runtime = [runtime_val]

        dtype_arg = None
        if len(node.args) > 1:
            dtype_arg = node.args[1]

        for kw in node.keywords:
            if kw.arg == "dtype":
                dtype_arg = kw.value
                break

        element_type = element_type_from_ast_node(dtype_arg, self.container_table)

        return self._create_array_temp(
            dims,
            element_type,
            zero_init=(func_name == "zeros"),
            ones_init=(func_name == "ones"),
            shapes_runtime=dims_runtime,
        )

    def _handle_numpy_empty_like(self, node, func_name):
        """Handle np.empty_like."""
        prototype_arg = node.args[0]
        prototype_name = self.visit(prototype_arg)

        dims = []
        if prototype_name in self.tensor_table:
            dims = self.tensor_table[prototype_name].shape

        dtype_arg = None
        if len(node.args) > 1:
            dtype_arg = node.args[1]

        for kw in node.keywords:
            if kw.arg == "dtype":
                dtype_arg = kw.value
                break

        element_type = None
        if dtype_arg:
            element_type = element_type_from_ast_node(dtype_arg, self.container_table)
        else:
            if prototype_name in self.container_table:
                sym_type = self.container_table[prototype_name]
                if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                    element_type = sym_type.pointee_type

        if element_type is None:
            element_type = Scalar(PrimitiveType.Double)

        return self._create_array_temp(
            dims, element_type, zero_init=False, ones_init=False
        )

    def _handle_numpy_zeros_like(self, node, func_name):
        """Handle np.zeros_like."""
        prototype_arg = node.args[0]
        prototype_name = self.visit(prototype_arg)

        dims = []
        if prototype_name in self.tensor_table:
            dims = self.tensor_table[prototype_name].shape

        dtype_arg = None
        if len(node.args) > 1:
            dtype_arg = node.args[1]

        for kw in node.keywords:
            if kw.arg == "dtype":
                dtype_arg = kw.value
                break

        element_type = None
        if dtype_arg:
            element_type = element_type_from_ast_node(dtype_arg, self.container_table)
        else:
            if prototype_name in self.container_table:
                sym_type = self.container_table[prototype_name]
                if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                    element_type = sym_type.pointee_type

        if element_type is None:
            element_type = Scalar(PrimitiveType.Double)

        return self._create_array_temp(
            dims, element_type, zero_init=True, ones_init=False
        )

    def _handle_numpy_eye(self, node, func_name):
        """Handle np.eye."""
        N_arg = node.args[0]
        N_str = self.visit(N_arg)

        M_str = N_str
        if len(node.args) > 1:
            M_str = self.visit(node.args[1])

        k_str = "0"
        if len(node.args) > 2:
            k_str = self.visit(node.args[2])

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

        element_type = element_type_from_ast_node(dtype_arg, self.container_table)

        ptr_name = self._create_array_temp([N_str, M_str], element_type, zero_init=True)

        loop_var = f"_i_{self._get_unique_id()}"
        if not self.builder.exists(loop_var):
            self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
            self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

        self.builder.begin_for(loop_var, "0", N_str, "1")

        cond = f"(({loop_var} + {k_str}) >= 0) & (({loop_var} + {k_str}) < {M_str})"
        self.builder.begin_if(cond)

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
        """Handle np.add, np.subtract, np.multiply, np.divide, etc."""
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
        return self.handle_array_binary_op(op_map[func_name], args[0], args[1])

    def _handle_numpy_unary_op(self, node, func_name):
        """Handle np.exp, np.sqrt, np.abs, etc."""
        args = [self.visit(arg) for arg in node.args]
        if len(args) != 1:
            raise NotImplementedError(f"Numpy function {func_name} requires 1 argument")

        op_name = func_name
        if op_name == "absolute":
            op_name = "abs"

        return self.handle_array_unary_op(op_name, args[0])

    def _handle_numpy_where(self, node, func_name):
        """Handle np.where(condition, x, y) - elementwise ternary selection."""
        if len(node.args) != 3:
            raise NotImplementedError("np.where requires 3 arguments (condition, x, y)")

        cond_name = self.visit(node.args[0])
        x_name = self.visit(node.args[1])
        y_name = self.visit(node.args[2])

        shape = []
        dtype = Scalar(PrimitiveType.Double)

        if cond_name in self.tensor_table:
            shape = self.tensor_table[cond_name].shape

        if not shape and y_name in self.tensor_table:
            shape = self.tensor_table[y_name].shape

        if not shape and x_name in self.tensor_table:
            shape = self.tensor_table[x_name].shape

        if not shape:
            raise NotImplementedError("np.where requires at least one array argument")

        if y_name in self.container_table:
            y_type = self.container_table[y_name]
            if isinstance(y_type, Pointer) and y_type.has_pointee_type():
                dtype = y_type.pointee_type
            elif isinstance(y_type, Scalar):
                dtype = y_type

        tmp_name = self._create_array_temp(shape, dtype)

        loop_vars = []
        for i, dim in enumerate(shape):
            loop_var = f"_where_i{i}_{self._get_unique_id()}"
            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)
            loop_vars.append(loop_var)
            self.builder.begin_for(loop_var, "0", str(dim), "1")

        linear_idx = self._compute_linear_index(loop_vars, shape, tmp_name, len(shape))

        cond_tmp = f"_where_cond_{self._get_unique_id()}"
        self.builder.add_container(cond_tmp, Scalar(PrimitiveType.Bool), False)
        self.container_table[cond_tmp] = Scalar(PrimitiveType.Bool)

        block_cond = self.builder.add_block()
        if cond_name in self.tensor_table:
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

        self.builder.begin_if(f"{cond_tmp} == true")

        block_true = self.builder.add_block()
        t_out_true = self.builder.add_access(block_true, tmp_name)
        if x_name in self.tensor_table:
            t_x = self.builder.add_access(block_true, x_name)
            t_task_true = self.builder.add_tasklet(
                block_true, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_true, t_x, "void", t_task_true, "_in", linear_idx
            )
        else:
            t_x, x_sub = self._add_read(block_true, x_name)
            t_task_true = self.builder.add_tasklet(
                block_true, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(block_true, t_x, "void", t_task_true, "_in", x_sub)
        self.builder.add_memlet(
            block_true, t_task_true, "_out", t_out_true, "void", linear_idx
        )

        self.builder.begin_else()

        block_false = self.builder.add_block()
        t_out_false = self.builder.add_access(block_false, tmp_name)
        if y_name in self.tensor_table:
            t_y = self.builder.add_access(block_false, y_name)
            t_task_false = self.builder.add_tasklet(
                block_false, TaskletCode.assign, ["_in"], ["_out"]
            )
            self.builder.add_memlet(
                block_false, t_y, "void", t_task_false, "_in", linear_idx
            )
        else:
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

        for _ in loop_vars:
            self.builder.end_for()

        return tmp_name

    def _handle_numpy_clip(self, node, func_name):
        """Handle np.clip(a, a_min, a_max) - elementwise clipping."""
        if len(node.args) != 3:
            raise NotImplementedError("np.clip requires 3 arguments (a, a_min, a_max)")

        arr_name = self.visit(node.args[0])
        a_min = self.visit(node.args[1])
        a_max = self.visit(node.args[2])

        tmp1 = self.handle_array_binary_op("max", arr_name, a_min)
        result = self.handle_array_binary_op("min", tmp1, a_max)

        return result

    def _handle_numpy_matmul(self, node, func_name):
        """Handle np.matmul, np.dot."""
        if len(node.args) != 2:
            raise NotImplementedError("matmul/dot requires 2 arguments")
        return self._handle_matmul_helper(node.args[0], node.args[1])

    def handle_numpy_matmul_op(self, left_node, right_node):
        """Handle the @ operator for matrix multiplication."""
        return self._handle_matmul_helper(left_node, right_node)

    def _handle_matmul_helper(self, left_node, right_node):
        """Helper for matrix multiplication operations."""
        res_a = self.parse_arg(left_node)
        res_b = self.parse_arg(right_node)

        if not res_a[0]:
            left_name = self.visit(left_node)
            left_node = ast.Name(id=left_name)
            res_a = self.parse_arg(left_node)

        if not res_b[0]:
            right_name = self.visit(right_node)
            right_node = ast.Name(id=right_name)
            res_b = self.parse_arg(right_node)

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

        dtype_a = self._ev._element_type(name_a)
        dtype_b = self._ev._element_type(name_b)
        dtype = promote_element_types(dtype_a, dtype_b)

        if is_scalar:
            tmp_name = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(tmp_name, dtype, False)
            self.container_table[tmp_name] = dtype
        else:
            tmp_name = self._create_array_temp(output_shape, dtype)

        if ndim_a > 2 or ndim_b > 2:
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

            self.handle_gemm(
                slice_c, ast.BinOp(left=slice_a, op=ast.MatMult(), right=slice_b)
            )

            for _ in range(batch_dims):
                self.builder.end_for()
        else:
            if is_scalar:
                self.handle_dot(
                    tmp_name,
                    ast.BinOp(left=left_node, op=ast.MatMult(), right=right_node),
                )
            else:
                self.handle_gemm(
                    tmp_name,
                    ast.BinOp(left=left_node, op=ast.MatMult(), right=right_node),
                )

        return tmp_name

    def _handle_numpy_outer(self, node, func_name):
        """Handle np.outer."""
        if len(node.args) != 2:
            raise NotImplementedError("outer requires 2 arguments")

        arg0 = node.args[0]
        arg1 = node.args[1]

        res_a = self.parse_arg(arg0)
        res_b = self.parse_arg(arg1)

        if not res_a[0]:
            left_name = self.visit(arg0)
            arg0 = ast.Name(id=left_name)
            res_a = self.parse_arg(arg0)

        if not res_b[0]:
            right_name = self.visit(arg1)
            arg1 = ast.Name(id=right_name)
            res_b = self.parse_arg(arg1)

        name_a, subset_a, shape_a, indices_a = res_a
        name_b, subset_b, shape_b, indices_b = res_b

        if not name_a or not name_b:
            raise NotImplementedError("Could not resolve outer operands")

        def get_flattened_size_expr(name, indices, shapes):
            size_expr = "1"
            for s in shapes:
                if size_expr == "1":
                    size_expr = str(s)
                else:
                    size_expr = f"({size_expr} * {str(s)})"
            return size_expr

        m_expr = get_flattened_size_expr(name_a, indices_a, shape_a)
        n_expr = get_flattened_size_expr(name_b, indices_b, shape_b)

        dtype_a = self._ev._element_type(name_a)
        dtype_b = self._ev._element_type(name_b)
        dtype = promote_element_types(dtype_a, dtype_b)

        tmp_name = self._create_array_temp([m_expr, n_expr], dtype)

        new_call_node = ast.Call(
            func=node.func, args=[arg0, arg1], keywords=node.keywords
        )

        self.handle_outer(tmp_name, new_call_node)

        return tmp_name

    def handle_ufunc_outer(self, node, ufunc_name):
        """Handle np.add.outer, np.subtract.outer, np.multiply.outer, etc."""
        if len(node.args) != 2:
            raise NotImplementedError(f"{ufunc_name}.outer requires 2 arguments")

        if ufunc_name == "multiply":
            return self._handle_numpy_outer(node, "outer")

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

        arg0 = node.args[0]
        arg1 = node.args[1]

        res_a = self.parse_arg(arg0)
        res_b = self.parse_arg(arg1)

        if not res_a[0]:
            left_name = self.visit(arg0)
            arg0 = ast.Name(id=left_name)
            res_a = self.parse_arg(arg0)

        if not res_b[0]:
            right_name = self.visit(arg1)
            arg1 = ast.Name(id=right_name)
            res_b = self.parse_arg(arg1)

        name_a, subset_a, shape_a, indices_a = res_a
        name_b, subset_b, shape_b, indices_b = res_b

        if not name_a or not name_b:
            raise NotImplementedError("Could not resolve ufunc outer operands")

        def get_flattened_size_expr(shapes):
            if not shapes:
                return "1"
            size_expr = str(shapes[0])
            for s in shapes[1:]:
                size_expr = f"({size_expr} * {str(s)})"
            return size_expr

        m_expr = get_flattened_size_expr(shape_a)
        n_expr = get_flattened_size_expr(shape_b)

        dtype_left = self._ev._element_type(name_a)
        dtype_right = self._ev._element_type(name_b)
        dtype = promote_element_types(dtype_left, dtype_right)

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

        tmp_name = self._create_array_temp([m_expr, n_expr], dtype)

        i_var = self.builder.find_new_name("_outer_i_")
        j_var = self.builder.find_new_name("_outer_j_")

        if not self.builder.exists(i_var):
            self.builder.add_container(i_var, Scalar(PrimitiveType.Int64), False)
            self.container_table[i_var] = Scalar(PrimitiveType.Int64)
        if not self.builder.exists(j_var):
            self.builder.add_container(j_var, Scalar(PrimitiveType.Int64), False)
            self.container_table[j_var] = Scalar(PrimitiveType.Int64)

        def compute_linear_index(name, subset, indices, loop_var):
            if not indices:
                return loop_var

            if name in self.tensor_table:
                info = self.tensor_table[name]
                shapes = info.shape
                ndim = len(shapes)
            else:
                shapes = []
                ndim = 0

            if ndim == 0:
                return loop_var

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

            terms = []
            loop_var_used = False

            for i, idx in enumerate(indices):
                stride = strides[i] if i < len(strides) else "1"
                start = subset[i] if i < len(subset) else "0"

                if isinstance(idx, ast.Slice):
                    if stride == "1":
                        term = f"({start} + {loop_var})"
                    else:
                        term = f"(({start} + {loop_var}) * {stride})"
                    loop_var_used = True
                else:
                    if stride == "1":
                        term = start
                    else:
                        term = f"({start} * {stride})"

                terms.append(term)

            if not terms:
                return loop_var

            result = terms[0]
            for t in terms[1:]:
                result = f"({result} + {t})"

            return result

        self.builder.begin_for(i_var, "0", m_expr, "1")
        self.builder.begin_for(j_var, "0", n_expr, "1")

        block = self.builder.add_block()

        t_a = self.builder.add_access(block, name_a)
        t_b = self.builder.add_access(block, name_b)
        t_c = self.builder.add_access(block, tmp_name)

        if ufunc_name in ["minimum", "maximum"]:
            if is_int:
                t_task = self.builder.add_tasklet(
                    block, int_opcode, ["_in1", "_in2"], ["_out"]
                )
            else:
                t_task = self.builder.add_cmath(block, fp_opcode)
        else:
            tasklet_code = int_opcode if is_int else fp_opcode
            t_task = self.builder.add_tasklet(
                block, tasklet_code, ["_in1", "_in2"], ["_out"]
            )

        a_index = compute_linear_index(name_a, subset_a, indices_a, i_var)
        b_index = compute_linear_index(name_b, subset_b, indices_b, j_var)

        self.builder.add_memlet(block, t_a, "void", t_task, "_in1", a_index)
        self.builder.add_memlet(block, t_b, "void", t_task, "_in2", b_index)

        flat_index = f"(({i_var}) * ({n_expr}) + ({j_var}))"
        self.builder.add_memlet(block, t_task, "_out", t_c, "void", flat_index)

        self.builder.end_for()
        self.builder.end_for()

        return tmp_name

    def _handle_numpy_reduce(self, node, func_name):
        """Handle np.sum, np.max, np.min, np.mean, np.std."""
        args = node.args
        keywords = {kw.arg: kw.value for kw in node.keywords}

        array_node = args[0]
        array_name = self.visit(array_node)

        if array_name not in self.tensor_table:
            raise ValueError(f"Reduction input must be an array, got {array_name}")

        input_shape = self.tensor_table[array_name].shape
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
        elif isinstance(axis, ast.Constant):
            val = axis.value
            if val < 0:
                val += ndim
            axes = [val]
        elif isinstance(axis, ast.Tuple):
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
            try:
                val = int(self.visit(axis))
                if val < 0:
                    val += ndim
                axes = [val]
            except:
                raise NotImplementedError("Dynamic axis not supported")

        output_shape = []
        for i in range(ndim):
            if i in axes:
                if keepdims:
                    output_shape.append("1")
            else:
                output_shape.append(input_shape[i])

        dtype = self._ev._element_type(array_name)

        if not output_shape:
            tmp_name = f"_tmp_{self._get_unique_id()}"
            self.builder.add_container(tmp_name, dtype, False)
            self.container_table[tmp_name] = dtype
            self.tensor_table[tmp_name] = Tensor(dtype, [])
        else:
            tmp_name = self._create_array_temp(output_shape, dtype)

        self.builder.add_reduce_op(
            func_name, array_name, tmp_name, input_shape, axes, keepdims
        )

        return tmp_name

    def handle_numpy_astype(self, node, array_name):
        """Handle numpy array.astype(dtype) method calls."""
        if len(node.args) < 1:
            raise ValueError("astype requires at least one argument (dtype)")

        dtype_arg = node.args[0]
        target_dtype = element_type_from_ast_node(dtype_arg, self.container_table)

        if array_name not in self.tensor_table:
            raise ValueError(f"Array {array_name} not found in tensor_table")

        input_shape = self.tensor_table[array_name].shape

        tmp_name = self._create_array_temp(input_shape, target_dtype)

        self.builder.add_cast_op(
            array_name, tmp_name, input_shape, target_dtype.primitive_type
        )

        return tmp_name

    def handle_numpy_copy(self, node, array_name):
        """Handle numpy array.copy() method calls using memcpy."""
        if array_name not in self.tensor_table:
            raise ValueError(f"Array {array_name} not found in tensor_table")

        input_shape = self.tensor_table[array_name].shape

        element_type = Scalar(PrimitiveType.Double)
        if array_name in self.container_table:
            sym_type = self.container_table[array_name]
            if isinstance(sym_type, Pointer) and sym_type.has_pointee_type():
                element_type = sym_type.pointee_type

        tmp_name = self._create_array_temp(input_shape, element_type)

        total_elements = " * ".join([f"({s})" for s in input_shape])
        element_size = self.builder.get_sizeof(element_type)
        count_expr = f"({total_elements}) * ({element_size})"

        ptr_type = Pointer(element_type)

        block = self.builder.add_block()
        t_src = self.builder.add_access(block, array_name)
        t_dst = self.builder.add_access(block, tmp_name)
        t_memcpy = self.builder.add_memcpy(block, count_expr)

        self.builder.add_memlet(block, t_src, "void", t_memcpy, "_src", "", ptr_type)
        self.builder.add_memlet(block, t_memcpy, "_dst", t_dst, "void", "", ptr_type)

        return tmp_name

    def _create_array_temp(
        self, shape, dtype, zero_init=False, ones_init=False, shapes_runtime=None
    ):
        """Create a temporary array with the given shape and dtype."""
        tmp_name = f"_tmp_{self._get_unique_id()}"

        # Handle 0-dimensional arrays as scalars
        if not shape or (len(shape) == 0):
            self.builder.add_container(tmp_name, dtype, False)
            self.container_table[tmp_name] = dtype
            self.tensor_table[tmp_name] = Tensor(dtype, [])

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
        self.container_table[tmp_name] = ptr_type
        tensor_entry = Tensor(dtype, shape)
        if shapes_runtime is not None:
            self.shapes_runtime_info[tmp_name] = shapes_runtime
        self.tensor_table[tmp_name] = tensor_entry

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
            loop_var = f"_i_{self._get_unique_id()}"
            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.container_table[loop_var] = Scalar(PrimitiveType.Int64)

            self.builder.begin_for(loop_var, "0", size_str, "1")

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

    def _compute_broadcast_shape(self, shape_a, shape_b):
        """Compute the broadcast output shape following NumPy broadcasting rules."""
        if not shape_a:
            return shape_b
        if not shape_b:
            return shape_a

        max_ndim = max(len(shape_a), len(shape_b))
        padded_a = ["1"] * (max_ndim - len(shape_a)) + [str(s) for s in shape_a]
        padded_b = ["1"] * (max_ndim - len(shape_b)) + [str(s) for s in shape_b]

        result = []
        for a, b in zip(padded_a, padded_b):
            if a == "1":
                result.append(b)
            elif b == "1":
                result.append(a)
            elif a == b:
                result.append(a)
            else:
                result.append(a)

        return result

    def _needs_broadcast(self, input_shape, output_shape):
        """Check if input shape needs broadcasting to match output shape."""
        if len(input_shape) != len(output_shape):
            return True
        for in_dim, out_dim in zip(input_shape, output_shape):
            if str(in_dim) != str(out_dim):
                return True
        return False

    def _broadcast_array(self, arr_name, input_shape, output_shape, dtype):
        """Broadcast an array from input_shape to output_shape using BroadcastNode."""
        broadcast_tmp = self._create_array_temp(output_shape, dtype)

        padded_input_shape = ["1"] * (len(output_shape) - len(input_shape)) + [
            str(s) for s in input_shape
        ]

        input_shape_strs = padded_input_shape
        output_shape_strs = [str(s) for s in output_shape]

        self.builder.add_broadcast(
            arr_name, broadcast_tmp, input_shape_strs, output_shape_strs
        )

        return broadcast_tmp

    def _shape_to_runtime_expr(self, shape_node):
        """Convert a shape expression AST node to a runtime-evaluable string."""
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
                return self.visit(shape_node)
        elif isinstance(shape_node, ast.Subscript):
            val = shape_node.value
            if isinstance(val, ast.Attribute) and val.attr == "shape":
                if isinstance(val.value, ast.Name):
                    arr_name = val.value.id
                    if isinstance(shape_node.slice, ast.Constant):
                        idx = shape_node.slice.value
                        if arr_name in self.tensor_table:
                            shapes = self.tensor_table[arr_name].shape
                            if idx < len(shapes):
                                return shapes[idx]
                        return f"{arr_name}.shape[{idx}]"
            return self.visit(shape_node)
        elif isinstance(shape_node, ast.Tuple):
            return [self._shape_to_runtime_expr(elt) for elt in shape_node.elts]
        elif isinstance(shape_node, ast.List):
            return [self._shape_to_runtime_expr(elt) for elt in shape_node.elts]
        else:
            return self.visit(shape_node)
