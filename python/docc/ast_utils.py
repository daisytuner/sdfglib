import ast
import copy
from ._sdfg import DebugInfo


def get_debug_info(node, filename, function_name=""):
    if hasattr(node, "lineno"):
        return DebugInfo(
            filename,
            function_name,
            node.lineno,
            node.col_offset + 1,
            (
                node.end_lineno
                if hasattr(node, "end_lineno") and node.end_lineno is not None
                else node.lineno
            ),
            (
                node.end_col_offset + 1
                if hasattr(node, "end_col_offset") and node.end_col_offset is not None
                else node.col_offset + 1
            ),
        )
    return DebugInfo()


class ArrayToElementRewriter(ast.NodeTransformer):
    def __init__(self, loop_vars, array_info):
        self.loop_vars = loop_vars
        self.array_info = array_info

    def visit_Name(self, node):
        if node.id in self.array_info:
            # Replace with subscript
            indices = [ast.Name(id=lv, ctx=ast.Load()) for lv in self.loop_vars]
            return ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=(
                    ast.Tuple(elts=indices, ctx=ast.Load())
                    if len(indices) > 1
                    else indices[0]
                ),
                ctx=ast.Load(),
            )
        return node


class SliceRewriter(ast.NodeTransformer):
    def __init__(self, loop_vars, array_info, expr_visitor):
        self.loop_vars = loop_vars
        self.array_info = array_info
        self.expr_visitor = expr_visitor

    def visit_Name(self, node):
        if node.id in self.array_info and self.loop_vars:
            ndim = self.array_info[node.id]["ndim"]
            if ndim == len(self.loop_vars):
                indices = [ast.Name(id=lv, ctx=ast.Load()) for lv in self.loop_vars]
                return ast.Subscript(
                    value=ast.Name(id=node.id, ctx=ast.Load()),
                    slice=(
                        ast.Tuple(elts=indices, ctx=ast.Load())
                        if len(indices) > 1
                        else indices[0]
                    ),
                    ctx=ast.Load(),
                )
        return node

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            if self.loop_vars:
                indices = [ast.Name(id=lv, ctx=ast.Load()) for lv in self.loop_vars]
                return ast.Subscript(
                    value=node,
                    slice=(
                        ast.Tuple(elts=indices, ctx=ast.Load())
                        if len(indices) > 1
                        else indices[0]
                    ),
                    ctx=ast.Load(),
                )
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)

        # We need to visit the value to get its string representation for array_info lookup
        # But expr_visitor.visit returns a string.
        # We assume expr_visitor can handle the node.
        value_str = self.expr_visitor.visit(node.value)
        if value_str not in self.array_info:
            return node

        indices = []
        if isinstance(node.slice, ast.Tuple):
            indices = node.slice.elts
        else:
            indices = [node.slice]

        ndim = self.array_info[value_str]["ndim"]
        if len(indices) < ndim:
            indices = list(indices)
            for _ in range(ndim - len(indices)):
                indices.append(ast.Slice(lower=None, upper=None, step=None))

        new_indices = []
        slice_counter = 0

        for i, idx in enumerate(indices):
            if isinstance(idx, ast.Slice):
                if slice_counter >= len(self.loop_vars):
                    raise ValueError("Rank mismatch in slice assignment")

                loop_var = self.loop_vars[slice_counter]
                slice_counter += 1

                start_str = "0"
                if idx.lower:
                    start_str = self.expr_visitor.visit(idx.lower)
                    if start_str.startswith("-"):
                        shapes = self.array_info[value_str].get("shapes", [])
                        dim_size = (
                            shapes[i] if i < len(shapes) else f"_{value_str}_shape_{i}"
                        )
                        start_str = f"({dim_size} {start_str})"

                step_str = "1"
                if idx.step:
                    step_str = self.expr_visitor.visit(idx.step)

                if step_str == "1":
                    if start_str == "0":
                        term = loop_var
                    else:
                        term = f"({start_str} + {loop_var})"
                else:
                    term = f"({start_str} + {loop_var} * {step_str})"
                new_indices.append(ast.Name(id=term, ctx=ast.Load()))
            else:
                new_indices.append(self.visit(idx))

        if len(new_indices) == 1:
            node.slice = new_indices[0]
        else:
            node.slice = ast.Tuple(elts=new_indices, ctx=ast.Load())

        return node


def get_unique_id(counter_ref):
    counter_ref[0] += 1
    return counter_ref[0]
