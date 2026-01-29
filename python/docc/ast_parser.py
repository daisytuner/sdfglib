import ast
import copy
from ._sdfg import Scalar, PrimitiveType, Pointer, TaskletCode
from .ast_utils import (
    SliceRewriter,
    get_debug_info,
    contains_ufunc_outer,
    normalize_negative_index,
)
from .expression_visitor import ExpressionVisitor
from .linear_algebra import LinearAlgebraHandler
from .convolution import ConvolutionHandler
from .onnx_ops import ONNXHandler


class ASTParser(ast.NodeVisitor):
    def __init__(
        self,
        builder,
        array_info=None,
        symbol_table=None,
        filename="",
        function_name="",
        infer_return_type=False,
        globals_dict=None,
        unique_counter_ref=None,
        structure_member_info=None,
    ):
        self.builder = builder
        self.array_info = array_info if array_info is not None else {}
        self.symbol_table = symbol_table if symbol_table is not None else {}
        self.filename = filename
        self.function_name = function_name
        self.infer_return_type = infer_return_type
        self.globals_dict = globals_dict
        self._unique_counter_ref = (
            unique_counter_ref if unique_counter_ref is not None else [0]
        )
        self.expr_visitor = ExpressionVisitor(
            self.array_info,
            self.builder,
            self.symbol_table,
            self.globals_dict,
            unique_counter_ref=self._unique_counter_ref,
            structure_member_info=structure_member_info,
        )
        self.la_handler = LinearAlgebraHandler(
            self.builder, self.array_info, self.symbol_table, self.expr_visitor
        )
        self.conv_handler = ConvolutionHandler(
            self.builder, self.array_info, self.symbol_table, self.expr_visitor
        )
        self.onnx_handler = ONNXHandler(
            self.builder, self.array_info, self.symbol_table, self.expr_visitor
        )
        self.expr_visitor.la_handler = self.la_handler
        self.captured_return_shapes = {}  # Map param name to shape string list

    def _get_unique_id(self):
        self._unique_counter_ref[0] += 1
        return self._unique_counter_ref[0]

    def _parse_expr(self, node):
        return self.expr_visitor.visit(node)

    def visit_Return(self, node):
        if node.value is None:
            debug_info = get_debug_info(node, self.filename, self.function_name)
            self.builder.add_return("", debug_info)
            return

        if isinstance(node.value, ast.Tuple):
            values = node.value.elts
        else:
            values = [node.value]

        parsed_values = [self._parse_expr(v) for v in values]
        debug_info = get_debug_info(node, self.filename, self.function_name)

        if self.infer_return_type:
            for i, res in enumerate(parsed_values):
                ret_name = f"_docc_ret_{i}"
                if not self.builder.exists(ret_name):
                    dtype = Scalar(PrimitiveType.Double)
                    if res in self.symbol_table:
                        dtype = self.symbol_table[res]
                    elif isinstance(values[i], ast.Constant):
                        val = values[i].value
                        if isinstance(val, int):
                            dtype = Scalar(PrimitiveType.Int64)
                        elif isinstance(val, float):
                            dtype = Scalar(PrimitiveType.Double)
                        elif isinstance(val, bool):
                            dtype = Scalar(PrimitiveType.Bool)

                    # Wrap Scalar in Pointer. Keep Arrays/Pointers as is.
                    arg_type = dtype
                    if isinstance(dtype, Scalar):
                        arg_type = Pointer(dtype)

                    self.builder.add_container(ret_name, arg_type, is_argument=True)
                    self.symbol_table[ret_name] = arg_type

                    if res in self.array_info:
                        self.array_info[ret_name] = self.array_info[res]

            self.infer_return_type = False

        for i, res in enumerate(parsed_values):
            ret_name = f"_docc_ret_{i}"
            typ = self.symbol_table.get(ret_name)

            is_array_return = False
            if res in self.array_info:
                # Only treat as array return if it has dimensions
                # 0-d arrays (scalars) should be handled by scalar assignment
                if self.array_info[res]["ndim"] > 0:
                    is_array_return = True
            elif res in self.symbol_table:
                if isinstance(self.symbol_table[res], Pointer):
                    is_array_return = True

            # Simple Scalar Assignment
            if not is_array_return:
                block = self.builder.add_block(debug_info)
                t_dst = self.builder.add_access(block, ret_name, debug_info)

                t_src, src_sub = self.expr_visitor._add_read(block, res, debug_info)

                t_task = self.builder.add_tasklet(
                    block, TaskletCode.assign, ["_in"], ["_out"], debug_info
                )
                self.builder.add_memlet(
                    block, t_src, "void", t_task, "_in", src_sub, None, debug_info
                )
                self.builder.add_memlet(
                    block, t_task, "_out", t_dst, "void", "0", None, debug_info
                )

            # Array Assignment (Copy)
            else:
                # Record shape for metadata
                if res in self.array_info:
                    # Prefer runtime shapes if available (for indirect access patterns)
                    # Fall back to regular shapes otherwise
                    if "shapes_runtime" in self.array_info[res]:
                        shape = self.array_info[res]["shapes_runtime"]
                    else:
                        shape = self.array_info[res]["shapes"]
                    # Convert to string expressions
                    self.captured_return_shapes[ret_name] = [str(s) for s in shape]

                    # Ensure destination array info exists
                    if ret_name not in self.array_info:
                        self.array_info[ret_name] = self.array_info[res]

                # Copy Logic using visit_Assign
                ndim = 1
                if ret_name in self.array_info:
                    ndim = self.array_info[ret_name]["ndim"]

                slice_node = ast.Slice(lower=None, upper=None, step=None)
                if ndim > 1:
                    target_slice = ast.Tuple(elts=[slice_node] * ndim, ctx=ast.Load())
                else:
                    target_slice = slice_node

                target_sub = ast.Subscript(
                    value=ast.Name(id=ret_name, ctx=ast.Load()),
                    slice=target_slice,
                    ctx=ast.Store(),
                )

                # Value node reconstruction
                if isinstance(values[i], ast.Name):
                    val_node = values[i]
                else:
                    val_node = ast.Name(id=res, ctx=ast.Load())

                assign_node = ast.Assign(targets=[target_sub], value=val_node)
                self.visit_Assign(assign_node)

        # Add control flow return to exit the function/path
        self.builder.add_return("", debug_info)

    def visit_Expr(self, node):
        """Handle expression statements (e.g., bare function calls)."""
        # Expression statements are typically function calls without assignment
        # Like: build_up_b(...) or pressure_poisson(...)
        if isinstance(node.value, ast.Call):
            # This is a function call statement - handle it through expression visitor
            # which will inline the function
            self._parse_expr(node.value)
        else:
            # For other expression statements, just evaluate them
            self._parse_expr(node.value)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self.array_info:
            # Convert to slice assignment: target[:] = target op value
            ndim = self.array_info[node.target.id]["ndim"]

            slices = []
            for _ in range(ndim):
                slices.append(ast.Slice(lower=None, upper=None, step=None))

            if ndim == 1:
                slice_arg = slices[0]
            else:
                slice_arg = ast.Tuple(elts=slices, ctx=ast.Load())

            slice_node = ast.Subscript(
                value=node.target, slice=slice_arg, ctx=ast.Store()
            )

            new_node = ast.Assign(
                targets=[slice_node],
                value=ast.BinOp(left=node.target, op=node.op, right=node.value),
            )
            self.visit_Assign(new_node)
        else:
            new_node = ast.Assign(
                targets=[node.target],
                value=ast.BinOp(left=node.target, op=node.op, right=node.value),
            )
            self.visit_Assign(new_node)

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            tmp_name = f"_assign_tmp_{self._get_unique_id()}"
            # Assign value to temporary
            val_assign = ast.Assign(
                targets=[ast.Name(id=tmp_name, ctx=ast.Store())], value=node.value
            )
            ast.copy_location(val_assign, node)
            self.visit_Assign(val_assign)

            # Assign temporary to targets
            for target in node.targets:
                assign = ast.Assign(
                    targets=[target], value=ast.Name(id=tmp_name, ctx=ast.Load())
                )
                ast.copy_location(assign, node)
                self.visit_Assign(assign)
            return

        target = node.targets[0]

        # Handle tuple unpacking: I, J, K = expr1, expr2, expr3
        if isinstance(target, ast.Tuple):
            if isinstance(node.value, ast.Tuple):
                # Unpacking tuple to tuple: a, b, c = x, y, z
                if len(target.elts) != len(node.value.elts):
                    raise ValueError("Tuple unpacking size mismatch")
                for tgt, val in zip(target.elts, node.value.elts):
                    assign = ast.Assign(targets=[tgt], value=val)
                    ast.copy_location(assign, node)
                    self.visit_Assign(assign)
            else:
                raise NotImplementedError(
                    "Tuple unpacking from non-tuple values not supported"
                )
            return

        # Special case: linear algebra functions
        if self.la_handler.is_gemm(node.value):
            if self.la_handler.handle_gemm(target, node.value):
                return
            if self.la_handler.handle_dot(target, node.value):
                return

        # Special case: outer product
        if self.la_handler.is_outer(node.value):
            if self.la_handler.handle_outer(target, node.value):
                return

        # Special case: convolution
        if self.conv_handler.is_conv(node.value):
            if self.conv_handler.handle_conv(target, node.value):
                return

        # Special case: ONNX ops (Transpose)
        if self.onnx_handler.is_transpose(node.value):
            if self.onnx_handler.handle_transpose(target, node.value):
                return

        # Special case:
        if isinstance(target, ast.Subscript):
            target_name = self.expr_visitor.visit(target.value)

            indices = []
            if isinstance(target.slice, ast.Tuple):
                indices = target.slice.elts
            else:
                indices = [target.slice]

            has_slice = False
            for idx in indices:
                if isinstance(idx, ast.Slice):
                    has_slice = True
                    break

            if has_slice:
                debug_info = get_debug_info(node, self.filename, self.function_name)
                self._handle_slice_assignment(
                    target, node.value, target_name, indices, debug_info
                )
                return

            target_name_full = self._parse_expr(target)
            value_str = self._parse_expr(node.value)
            debug_info = get_debug_info(node, self.filename, self.function_name)

            block = self.builder.add_block(debug_info)
            t_src, src_sub = self.expr_visitor._add_read(block, value_str, debug_info)

            if "(" in target_name_full and target_name_full.endswith(")"):
                name = target_name_full.split("(")[0]
                subset = target_name_full[target_name_full.find("(") + 1 : -1]
                t_dst = self.builder.add_access(block, name, debug_info)
                dst_sub = subset
            else:
                t_dst = self.builder.add_access(block, target_name_full, debug_info)
                dst_sub = ""

            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )

            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", src_sub, None, debug_info
            )
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", dst_sub, None, debug_info
            )
            return

        # Variable assignments
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only assignment to variables supported")

        target_name = target.id
        value_str = self._parse_expr(node.value)
        debug_info = get_debug_info(node, self.filename, self.function_name)

        if not self.builder.exists(target_name):
            if isinstance(node.value, ast.Constant):
                val = node.value.value
                if isinstance(val, int):
                    dtype = Scalar(PrimitiveType.Int64)
                elif isinstance(val, float):
                    dtype = Scalar(PrimitiveType.Double)
                elif isinstance(val, bool):
                    dtype = Scalar(PrimitiveType.Bool)
                else:
                    raise NotImplementedError(f"Cannot infer type for {val}")

                self.builder.add_container(target_name, dtype, False)
                self.symbol_table[target_name] = dtype
            else:
                assert value_str in self.symbol_table
                self.builder.add_container(
                    target_name, self.symbol_table[value_str], False
                )
                self.symbol_table[target_name] = self.symbol_table[value_str]

        if value_str in self.array_info:
            self.array_info[target_name] = self.array_info[value_str]

        # Distinguish assignments: scalar -> tasklet, pointer -> reference_memlet
        src_type = self.symbol_table.get(value_str)
        dst_type = self.symbol_table[target_name]
        if src_type and isinstance(src_type, Pointer) and isinstance(dst_type, Pointer):
            block = self.builder.add_block(debug_info)
            t_src = self.builder.add_access(block, value_str, debug_info)
            t_dst = self.builder.add_access(block, target_name, debug_info)
            self.builder.add_reference_memlet(
                block, t_src, t_dst, "0", src_type, debug_info
            )
            return
        elif (src_type and isinstance(src_type, Scalar)) or isinstance(
            dst_type, Scalar
        ):
            block = self.builder.add_block(debug_info)
            t_dst = self.builder.add_access(block, target_name, debug_info)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )

            if src_type:
                t_src = self.builder.add_access(block, value_str, debug_info)
            else:
                t_src = self.builder.add_constant(
                    block, value_str, dst_type, debug_info
                )

            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", "", None, debug_info
            )
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", "", None, debug_info
            )

            return

    def visit_If(self, node):
        cond = self._parse_expr(node.test)
        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.begin_if(f"{cond} != false", debug_info)

        for stmt in node.body:
            self.visit(stmt)

        if node.orelse:
            self.builder.begin_else(debug_info)
            for stmt in node.orelse:
                self.visit(stmt)

        self.builder.end_if()

    def visit_While(self, node):
        cond = self._parse_expr(node.test)
        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.begin_while(f"{cond} != false", debug_info)

        for stmt in node.body:
            self.visit(stmt)

        self.builder.end_while()

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple for loops supported")

        var = node.target.id

        if not isinstance(node.iter, ast.Call) or node.iter.func.id != "range":
            raise NotImplementedError("Only range() loops supported")

        args = node.iter.args
        if len(args) == 1:
            start = "0"
            end = self._parse_expr(args[0])
            step = "1"
        elif len(args) == 2:
            start = self._parse_expr(args[0])
            end = self._parse_expr(args[1])
            step = "1"
        elif len(args) == 3:
            start = self._parse_expr(args[0])
            end = self._parse_expr(args[1])

            # Special handling for step to avoid creating tasklets for constants
            step_node = args[2]
            if isinstance(step_node, ast.Constant):
                step = str(step_node.value)
            elif (
                isinstance(step_node, ast.UnaryOp)
                and isinstance(step_node.op, ast.USub)
                and isinstance(step_node.operand, ast.Constant)
            ):
                step = f"-{step_node.operand.value}"
            else:
                step = self._parse_expr(step_node)
        else:
            raise ValueError("Invalid range arguments")

        if not self.builder.exists(var):
            self.builder.add_container(var, Scalar(PrimitiveType.Int64), False)
            self.symbol_table[var] = Scalar(PrimitiveType.Int64)

        debug_info = get_debug_info(node, self.filename, self.function_name)
        self.builder.begin_for(var, start, end, step, debug_info)

        for stmt in node.body:
            self.visit(stmt)

        self.builder.end_for()

    def _get_max_array_ndim_in_expr(self, node):
        """Get the maximum array dimensionality in an expression."""
        max_ndim = 0

        class NdimVisitor(ast.NodeVisitor):
            def __init__(self, array_info):
                self.array_info = array_info
                self.max_ndim = 0

            def visit_Name(self, node):
                if node.id in self.array_info:
                    ndim = self.array_info[node.id].get("ndim", 0)
                    self.max_ndim = max(self.max_ndim, ndim)
                return self.generic_visit(node)

        visitor = NdimVisitor(self.array_info)
        visitor.visit(node)
        return visitor.max_ndim

    def _handle_broadcast_slice_assignment(
        self, target, value, target_name, indices, target_ndim, value_ndim, debug_info
    ):
        """Handle slice assignment with broadcasting (e.g., 2D -= 1D)."""
        # Number of broadcast dimensions (outer loops)
        broadcast_dims = target_ndim - value_ndim

        shapes = self.array_info[target_name].get("shapes", [])

        # Create outer loops for broadcast dimensions
        outer_loop_vars = []
        for i in range(broadcast_dims):
            loop_var = f"_bcast_iter_{i}_{self._get_unique_id()}"
            outer_loop_vars.append(loop_var)

            if not self.builder.exists(loop_var):
                self.builder.add_container(loop_var, Scalar(PrimitiveType.Int64), False)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

            dim_size = shapes[i] if i < len(shapes) else f"_{target_name}_shape_{i}"
            self.builder.begin_for(loop_var, "0", dim_size, "1", debug_info)

        # Create a row view (reference) for the inner dimensions
        row_view_name = f"_row_view_{self._get_unique_id()}"

        # Get inner shape for the row view
        inner_shapes = shapes[broadcast_dims:] if len(shapes) > broadcast_dims else []

        # Determine element type from the target
        target_type = self.symbol_table.get(target_name)
        if isinstance(target_type, Pointer) and target_type.has_pointee_type():
            element_type = target_type.pointee_type
        else:
            element_type = Scalar(PrimitiveType.Double)

        # Create pointer type for row view
        row_type = Pointer(element_type)
        self.builder.add_container(row_view_name, row_type, False)
        self.symbol_table[row_view_name] = row_type

        # Register row view in array_info
        self.array_info[row_view_name] = {"ndim": value_ndim, "shapes": inner_shapes}

        # Create reference memlet: row_view = &target[i, 0, 0, ...]
        # The index is: outer_loop_vars joined, then zeros for inner dims
        ref_index_parts = outer_loop_vars[:]
        for _ in range(value_ndim):
            ref_index_parts.append("0")

        # Compute linearized index for reference
        # For target[i, j] with shape (n, m), linear index for row i is i * m
        linear_idx = outer_loop_vars[0] if outer_loop_vars else "0"
        for dim_idx in range(1, broadcast_dims):
            dim_size = (
                shapes[dim_idx]
                if dim_idx < len(shapes)
                else f"_{target_name}_shape_{dim_idx}"
            )
            linear_idx = f"({linear_idx}) * ({dim_size}) + {outer_loop_vars[dim_idx]}"

        # Multiply by inner dimension sizes to get the start of the row
        for dim_idx in range(broadcast_dims, target_ndim):
            dim_size = (
                shapes[dim_idx]
                if dim_idx < len(shapes)
                else f"_{target_name}_shape_{dim_idx}"
            )
            linear_idx = f"({linear_idx}) * ({dim_size})"

        # Create the reference memlet block
        block = self.builder.add_block(debug_info)
        t_src = self.builder.add_access(block, target_name, debug_info)
        t_dst = self.builder.add_access(block, row_view_name, debug_info)
        self.builder.add_reference_memlet(
            block, t_src, t_dst, linear_idx, row_type, debug_info
        )

        # Now handle the inner slice assignment with the row view
        # Create inner indices (all slices for the inner dimensions)
        inner_indices = [
            ast.Slice(lower=None, upper=None, step=None) for _ in range(value_ndim)
        ]

        # Create new target using row view
        new_target = ast.Subscript(
            value=ast.Name(id=row_view_name, ctx=ast.Load()),
            slice=(
                ast.Tuple(elts=inner_indices, ctx=ast.Load())
                if len(inner_indices) > 1
                else inner_indices[0]
            ),
            ctx=ast.Store(),
        )

        # Recursively handle the inner assignment (now same-dimension)
        self._handle_slice_assignment(
            new_target, value, row_view_name, inner_indices, debug_info
        )

        # Close outer loops
        for _ in outer_loop_vars:
            self.builder.end_for()

    def _handle_slice_assignment(
        self, target, value, target_name, indices, debug_info=None
    ):
        if debug_info is None:
            debug_info = DebugInfo()

        if target_name in self.array_info:
            ndim = self.array_info[target_name]["ndim"]
            if len(indices) < ndim:
                indices = list(indices)
                for _ in range(ndim - len(indices)):
                    indices.append(ast.Slice(lower=None, upper=None, step=None))

        # Check if the RHS contains a ufunc outer operation
        # If so, we handle it specially to avoid the loop transformation
        # which would destroy the slice shape information
        has_outer, ufunc_name, outer_node = contains_ufunc_outer(value)
        if has_outer:
            self._handle_ufunc_outer_slice_assignment(
                target, value, target_name, indices, debug_info
            )
            return

        # Count slice dimensions to determine effective target dimensionality
        # (slice indices produce array dimensions, point indices collapse them)
        target_slice_ndim = sum(1 for idx in indices if isinstance(idx, ast.Slice))
        value_max_ndim = self._get_max_array_ndim_in_expr(value)

        if (
            target_slice_ndim > 0
            and value_max_ndim > 0
            and target_slice_ndim > value_max_ndim
        ):
            # Broadcasting case: use row-by-row approach with reference memlets
            self._handle_broadcast_slice_assignment(
                target,
                value,
                target_name,
                indices,
                target_slice_ndim,
                value_max_ndim,
                debug_info,
            )
            return

        loop_vars = []
        new_target_indices = []

        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                loop_var = f"_slice_iter_{len(loop_vars)}_{self._get_unique_id()}"
                loop_vars.append(loop_var)

                if not self.builder.exists(loop_var):
                    self.builder.add_container(
                        loop_var, Scalar(PrimitiveType.Int64), False
                    )
                    self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

                start_str = "0"
                if idx.lower:
                    start_str = self._parse_expr(idx.lower)
                    if start_str.startswith("-"):
                        shapes = self.array_info[target_name].get("shapes", [])
                        dim_size = (
                            str(shapes[i])
                            if i < len(shapes)
                            else f"_{target_name}_shape_{i}"
                        )
                        start_str = f"({dim_size} {start_str})"

                stop_str = ""
                if idx.upper and not (
                    isinstance(idx.upper, ast.Constant) and idx.upper.value is None
                ):
                    stop_str = self._parse_expr(idx.upper)
                    if stop_str.startswith("-") or stop_str.startswith("(-"):
                        shapes = self.array_info[target_name].get("shapes", [])
                        dim_size = (
                            str(shapes[i])
                            if i < len(shapes)
                            else f"_{target_name}_shape_{i}"
                        )
                        stop_str = f"({dim_size} {stop_str})"
                else:
                    shapes = self.array_info[target_name].get("shapes", [])
                    stop_str = (
                        str(shapes[i])
                        if i < len(shapes)
                        else f"_{target_name}_shape_{i}"
                    )

                step_str = "1"
                if idx.step:
                    step_str = self._parse_expr(idx.step)

                count_str = f"({stop_str} - {start_str})"

                self.builder.begin_for(loop_var, "0", count_str, "1", debug_info)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

                new_target_indices.append(
                    ast.Name(
                        id=f"{start_str} + {loop_var} * {step_str}", ctx=ast.Load()
                    )
                )
            else:
                # Handle non-slice indices - need to normalize negative indices
                shapes = self.array_info[target_name].get("shapes", [])
                dim_size = shapes[i] if i < len(shapes) else f"_{target_name}_shape_{i}"
                normalized_idx = normalize_negative_index(idx, dim_size)
                new_target_indices.append(normalized_idx)

        rewriter = SliceRewriter(loop_vars, self.array_info, self.expr_visitor)
        new_value = rewriter.visit(copy.deepcopy(value))

        new_target = copy.deepcopy(target)
        if len(new_target_indices) == 1:
            new_target.slice = new_target_indices[0]
        else:
            new_target.slice = ast.Tuple(elts=new_target_indices, ctx=ast.Load())

        target_str = self._parse_expr(new_target)
        value_str = self._parse_expr(new_value)
        self.builder.add_assignment(target_str, value_str, debug_info)

        for _ in loop_vars:
            self.builder.end_for()

    def _handle_ufunc_outer_slice_assignment(
        self, target, value, target_name, indices, debug_info=None
    ):
        """Handle slice assignment where RHS contains a ufunc outer operation.

        Example: path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))

        The strategy is:
        1. Evaluate the entire RHS expression, which will create a temporary array
           containing the result of the ufunc outer (potentially wrapped in other ops)
        2. Copy the temporary result to the target slice

        This avoids the loop transformation that would destroy slice shape info.
        """
        if debug_info is None:
            from ._sdfg import DebugInfo

            debug_info = DebugInfo()

        # Evaluate the full RHS expression
        # This will:
        # - Create temp arrays for ufunc outer results
        # - Apply any wrapping operations (np.minimum, etc.)
        # - Return the name of the final result array
        result_name = self._parse_expr(value)

        # Now we need to copy result to target slice
        # Count slice dimensions to determine if we need loops
        target_slice_ndim = sum(1 for idx in indices if isinstance(idx, ast.Slice))

        if target_slice_ndim == 0:
            # No slices on target - just simple assignment
            target_str = self._parse_expr(target)
            block = self.builder.add_block(debug_info)
            t_src, src_sub = self.expr_visitor._add_read(block, result_name, debug_info)
            t_dst = self.builder.add_access(block, target_str, debug_info)
            t_task = self.builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )
            self.builder.add_memlet(
                block, t_src, "void", t_task, "_in", src_sub, None, debug_info
            )
            self.builder.add_memlet(
                block, t_task, "_out", t_dst, "void", "", None, debug_info
            )
            return

        # We have slices on the target - need to create loops for copying
        # Get target array info
        target_info = self.array_info.get(target_name, {})
        target_shapes = target_info.get("shapes", [])

        loop_vars = []
        new_target_indices = []

        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                loop_var = f"_copy_iter_{len(loop_vars)}_{self._get_unique_id()}"
                loop_vars.append(loop_var)

                if not self.builder.exists(loop_var):
                    self.builder.add_container(
                        loop_var, Scalar(PrimitiveType.Int64), False
                    )
                    self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

                start_str = "0"
                if idx.lower:
                    start_str = self._parse_expr(idx.lower)

                stop_str = ""
                if idx.upper and not (
                    isinstance(idx.upper, ast.Constant) and idx.upper.value is None
                ):
                    stop_str = self._parse_expr(idx.upper)
                else:
                    stop_str = (
                        target_shapes[i]
                        if i < len(target_shapes)
                        else f"_{target_name}_shape_{i}"
                    )

                step_str = "1"
                if idx.step:
                    step_str = self._parse_expr(idx.step)

                count_str = f"({stop_str} - {start_str})"

                self.builder.begin_for(loop_var, "0", count_str, "1", debug_info)
                self.symbol_table[loop_var] = Scalar(PrimitiveType.Int64)

                new_target_indices.append(
                    ast.Name(
                        id=f"{start_str} + {loop_var} * {step_str}", ctx=ast.Load()
                    )
                )
            else:
                # Handle non-slice indices - need to normalize negative indices
                dim_size = (
                    target_shapes[i]
                    if i < len(target_shapes)
                    else f"_{target_name}_shape_{i}"
                )
                normalized_idx = normalize_negative_index(idx, dim_size)
                new_target_indices.append(normalized_idx)

        # Create assignment block: target[i,j,...] = result[i,j,...]
        block = self.builder.add_block(debug_info)

        # Access nodes
        t_src = self.builder.add_access(block, result_name, debug_info)
        t_dst = self.builder.add_access(block, target_name, debug_info)
        t_task = self.builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )

        # Source index - just use loop vars for flat array from ufunc outer
        # The ufunc outer result is a flat array of size M*N
        if len(loop_vars) == 2:
            # 2D case: result is indexed as i * N + j
            # Get the second dimension size from target shapes
            n_dim = (
                target_shapes[1]
                if len(target_shapes) > 1
                else f"_{target_name}_shape_1"
            )
            src_index = f"(({loop_vars[0]}) * ({n_dim}) + ({loop_vars[1]}))"
        elif len(loop_vars) == 1:
            src_index = loop_vars[0]
        else:
            # General case - compute linear index
            src_terms = []
            stride = "1"
            for i in range(len(loop_vars) - 1, -1, -1):
                if stride == "1":
                    src_terms.insert(0, loop_vars[i])
                else:
                    src_terms.insert(0, f"({loop_vars[i]} * {stride})")
                if i > 0:
                    dim_size = (
                        target_shapes[i]
                        if i < len(target_shapes)
                        else f"_{target_name}_shape_{i}"
                    )
                    stride = (
                        f"({stride} * {dim_size})" if stride != "1" else str(dim_size)
                    )
            src_index = " + ".join(src_terms) if src_terms else "0"

        # Target index - compute linear index (row-major order)
        # For 2D array with shape (M, N): linear_index = i * N + j
        target_index_parts = []
        for idx in new_target_indices:
            if isinstance(idx, ast.Name):
                target_index_parts.append(idx.id)
            else:
                target_index_parts.append(self._parse_expr(idx))

        # Convert to linear index
        if len(target_index_parts) == 2:
            # 2D case
            n_dim = (
                target_shapes[1]
                if len(target_shapes) > 1
                else f"_{target_name}_shape_1"
            )
            target_index = (
                f"(({target_index_parts[0]}) * ({n_dim}) + ({target_index_parts[1]}))"
            )
        elif len(target_index_parts) == 1:
            target_index = target_index_parts[0]
        else:
            # General case - compute linear index with strides
            stride = "1"
            target_index = "0"
            for i in range(len(target_index_parts) - 1, -1, -1):
                idx_part = target_index_parts[i]
                if stride == "1":
                    term = idx_part
                else:
                    term = f"(({idx_part}) * ({stride}))"

                if target_index == "0":
                    target_index = term
                else:
                    target_index = f"({term} + {target_index})"

                if i > 0:
                    dim_size = (
                        target_shapes[i]
                        if i < len(target_shapes)
                        else f"_{target_name}_shape_{i}"
                    )
                    stride = (
                        f"({stride} * {dim_size})" if stride != "1" else str(dim_size)
                    )

        # Connect memlets
        self.builder.add_memlet(
            block, t_src, "void", t_task, "_in", src_index, None, debug_info
        )
        self.builder.add_memlet(
            block, t_task, "_out", t_dst, "void", target_index, None, debug_info
        )

        # End loops
        for _ in loop_vars:
            self.builder.end_for()
