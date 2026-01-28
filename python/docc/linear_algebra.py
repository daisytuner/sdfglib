import ast
from ._sdfg import Scalar, PrimitiveType


class LinearAlgebraHandler:
    def __init__(self, builder, array_info, symbol_table, expr_visitor):
        self.builder = builder
        self.array_info = array_info
        self.symbol_table = symbol_table
        self.expr_visitor = expr_visitor
        self._unique_counter = 0

    def _get_unique_id(self):
        self._unique_counter += 1
        return self._unique_counter

    def _parse_expr(self, node):
        return self.expr_visitor.visit(node)

    def is_gemm(self, node):
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

    def parse_arg(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.array_info:
                return node.id, [], self.array_info[node.id]["shapes"], []
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.array_info:
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
                            start = self._parse_expr(idx.lower)
                        start_indices.append(start)

                        shapes = self.array_info[name]["shapes"]
                        dim_size = (
                            shapes[i] if i < len(shapes) else f"_{name}_shape_{i}"
                        )
                        stop = dim_size
                        if idx.upper:
                            stop = self._parse_expr(idx.upper)

                        size = f"({stop} - {start})"
                        slice_shape.append(size)
                    else:
                        if isinstance(idx, ast.Name) and idx.id in self.array_info:
                            # This is an array index (gather operation)
                            return None, None, None, None
                        val = self._parse_expr(idx)
                        start_indices.append(val)

                return name, start_indices, slice_shape, indices

        return None, None, None, None

    def flatten_subset(self, name, start_indices):
        if not start_indices:
            return []
        info = self.array_info[name]
        shapes = info["shapes"]
        ndim = info["ndim"]

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

    def handle_gemm(self, target, value_node):
        target_name = None
        target_subset = []

        if isinstance(target, str):
            target_name = target
        elif isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name):
                # Handle target slice
                res = self.parse_arg(target)
                if res[0]:
                    target_name = res[0]
                    target_subset = self.flatten_subset(target_name, res[1])
                else:
                    target_name = target.value.id

        if not target_name or target_name not in self.array_info:
            return False

        alpha = "1.0"
        beta = "0.0"
        A = None
        B = None

        def extract_factor(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if self.is_gemm(node.left):
                    return node.left, self._parse_expr(node.right)
                if self.is_gemm(node.right):
                    return node.right, self._parse_expr(node.left)

                res = self.parse_arg(node.left)
                if res[0]:
                    return node.left, self._parse_expr(node.right)
                res = self.parse_arg(node.right)
                if res[0]:
                    return node.right, self._parse_expr(node.left)
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
                    return l, r, self._parse_expr(node.right)
                l, r, a = parse_term(node.right)
                if l:
                    return l, r, self._parse_expr(node.left)

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
                        beta = self._parse_expr(value_node.right.right)
                    elif self._is_target(value_node.right.right, target_name):
                        beta = self._parse_expr(value_node.right.left)
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
                            beta = self._parse_expr(value_node.left.right)
                        elif self._is_target(value_node.left.right, target_name):
                            beta = self._parse_expr(value_node.left.left)
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
            tmp_name = self.expr_visitor.visit(A_node)
            A_node = ast.Name(id=tmp_name)

        if self.is_gemm(B_node):
            tmp_name = self.expr_visitor.visit(B_node)
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
            if name not in self.array_info:
                return 1
            return self.array_info[name]["ndim"]

        if len(shape_a) == 2:
            if not trans_a:
                m = shape_a[0]
                k = shape_a[1]
            else:
                m = shape_a[1]
                k = shape_a[0]
        else:
            # 1D array A(K) -> (1, K)
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
            # 1D array B(K) -> (K, 1)
            n = "1"
            if self._is_stride_1(B_name, indices_b):
                if get_ndim(B_name) == 1:
                    trans_b = False
                else:
                    trans_b = True
            else:
                trans_b = False

        def get_ld(name):
            if name not in self.array_info:
                return ""
            shapes = self.array_info[name]["shapes"]
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

    def _is_stride_1(self, name, indices):
        if name not in self.array_info:
            return True
        info = self.array_info[name]
        ndim = info["ndim"]

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
        if isinstance(target_name, ast.AST):
            return self._parse_expr(node) == self._parse_expr(target_name)

        if isinstance(node, ast.Name) and node.id == target_name:
            return True
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == target_name:
                return True
        return False

    def _is_dot_call(self, node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "dot":
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        return False

    def handle_dot(self, target, value_node):
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
            info = self.array_info[name]
            shapes = info["shapes"]
            ndim = info["ndim"]

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
        self.symbol_table[tmp_res] = Scalar(PrimitiveType.Double)

        self.builder.add_dot(
            name_a, name_b, tmp_res, n, incx, incy, flat_subset_a, flat_subset_b
        )

        target_str = target if isinstance(target, str) else self._parse_expr(target)

        # Ensure target container exists for new scalar variables
        if not self.builder.exists(target_str):
            self.builder.add_container(target_str, Scalar(PrimitiveType.Double), False)
            self.symbol_table[target_str] = Scalar(PrimitiveType.Double)

        if is_accumulate:
            self.builder.add_assignment(target_str, f"{target_str} + {tmp_res}")
        else:
            self.builder.add_assignment(target_str, tmp_res)

        return True

    def is_outer(self, node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "outer":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "outer":
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.is_outer(node.left) or self.is_outer(node.right)
        return False

    def handle_outer(self, target, value_node):
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

        # Determine shapes M and N
        # outer(a, b) -> a is flattened to (M,), b is flattened to (N,)
        # result is (M, N)

        # We need to compute size of M and N from shapes and subsets if sliced.
        def get_flattened_size(name, indices, shapes):
            size_expr = "1"
            for s in shapes:
                if size_expr == "1":
                    size_expr = str(s)
                else:
                    size_expr = f"({size_expr} * {str(s)})"
            return size_expr

        def get_ld_2d(name):
            if name in self.array_info:
                shapes = self.array_info[name]["shapes"]
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

            # A: (M, 1) Column Vector. Use as is.
            # B: (N, 1) Column Vector. Transpose to get (1, N).
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
