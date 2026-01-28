import ast
from ._sdfg import Scalar, PrimitiveType, Pointer
from .ast_utils import get_debug_info


class ConvolutionHandler:
    def __init__(self, builder, array_info, symbol_table, expr_visitor):
        self.builder = builder
        self.array_info = array_info
        self.symbol_table = symbol_table
        self.expr_visitor = expr_visitor

    def _parse_expr(self, node):
        return self.expr_visitor.visit(node)

    def is_conv(self, node):
        if not isinstance(node, ast.Call):
            return False

        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "correlate2d":
                return True
        elif isinstance(node.func, ast.Name):
            if node.func.id == "correlate2d":
                return True

        return False

    def handle_conv(self, target, value_node):
        if not self.is_conv(value_node):
            return False

        args = value_node.args
        if len(args) < 2:
            return False

        in1_node = args[0]
        in2_node = args[1]

        in1_name = self._parse_expr(in1_node)
        in2_name = self._parse_expr(in2_node)

        if in1_name not in self.array_info:
            return False
        if in2_name not in self.array_info:
            return False

        in1_info = self.array_info[in1_name]
        in2_info = self.array_info[in2_name]

        # Check dimensions
        if in1_info["ndim"] != 2 or in2_info["ndim"] != 2:
            raise NotImplementedError(
                "Only 2D convolution is currently supported via scipy.signal mapping"
            )

        in1_shape = in1_info["shapes"]
        in2_shape = in2_info["shapes"]

        # Scipy Correlate2d / Convolve2d
        # Default mode is 'full', boundary 'fill', fillvalue 0

        mode = "full"
        # Parse kwargs
        for keyword in value_node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = keyword.value.value

        # Also check positional args for mode
        if len(args) > 2 and isinstance(args[2], ast.Constant):
            mode = args[2].value

        if mode != "valid" and mode != "full" and mode != "same":
            raise NotImplementedError(f"Unsupported convolution mode: {mode}")

        # Map to ConvNode
        # Treat as N=1, C_in=1, C_out=1

        shape_strs = ["1", "1"] + [str(s) for s in in1_shape]
        kernel_shape_strs = [str(s) for s in in2_shape]

        # Default strides 1
        strides = ["1", "1"]
        dilations = ["1", "1"]
        group = "1"
        output_channels = "1"

        pads = ["0", "0", "0", "0"]

        if mode == "valid":
            pads = ["0", "0", "0", "0"]
        elif mode == "full":
            # Padding is kernel_size - 1 on both sides
            # shapes are symbolic strings, so we construct the padding string
            # This is tricky without a symbolic engine in Python.
            # But we can produce a string expression that SDFG builder parses.
            h_k = kernel_shape_strs[0]
            w_k = kernel_shape_strs[1]
            pad_h = f"({h_k} - 1)"
            pad_w = f"({w_k} - 1)"
            pads = [pad_h, pad_w, pad_h, pad_w]
        elif mode == "same":
            # Padding is kernel_size // 2
            h_k = kernel_shape_strs[0]
            w_k = kernel_shape_strs[1]
            pad_h = f"idiv({h_k}, 2)"
            pad_w = f"idiv({w_k}, 2)"
            pads = [pad_h, pad_w, pad_h, pad_w]

        target_name = ""
        if isinstance(target, ast.Name):
            target_name = target.id
        elif isinstance(target, str):
            target_name = target

        if not target_name:
            return False

        if self.builder.exists(target_name):
            # Ensure shape is inferred
            pass
        else:
            # Infer shape
            out_shape = []
            H1 = str(in1_shape[0])
            W1 = str(in1_shape[1])
            H2 = str(in2_shape[0])
            W2 = str(in2_shape[1])

            if mode == "valid":
                out_shape = [f"({H1} - {H2} + 1)", f"({W1} - {W2} + 1)"]
            elif mode == "same":
                out_shape = [H1, W1]
            elif mode == "full":
                out_shape = [f"({H1} + {H2} - 1)", f"({W1} + {W2} - 1)"]

            # Use Double type (float)
            dtype = Scalar(PrimitiveType.Double)
            ptr_type = Pointer(dtype)

            self.builder.add_container(target_name, ptr_type, False)

            # Update parser state
            self.symbol_table[target_name] = ptr_type
            self.array_info[target_name] = {"ndim": 2, "shapes": out_shape}

            # Allocate memory for the result
            block_alloc = self.builder.add_block()

            # Calculate size: shape[0] * shape[1] * sizeof(double)
            # Assuming double (8 bytes)
            size_expr = f"(({out_shape[0]}) * ({out_shape[1]}))"
            total_size_expr = f"({size_expr} * 8)"

            t_malloc = self.builder.add_malloc(block_alloc, total_size_expr)
            t_ptr = self.builder.add_access(block_alloc, target_name)
            self.builder.add_memlet(
                block_alloc, t_malloc, "_ret", t_ptr, "void", "", ptr_type
            )

        debug_info = get_debug_info(
            value_node, getattr(self.builder, "filename", ""), ""
        )
        # Note: filename might not be accessible easily if not passed to handler.
        # But ASTParser passes filename to debug info helpers usually.
        # We'll pass a generic debug info if needed or modify init.

        # wait, ASTParser initializes Handler.

        self.builder.add_conv(
            in1_name,
            in2_name,
            target_name,
            shape_strs,
            kernel_shape_strs,
            strides,
            pads,
            dilations,
            output_channels,
            group,
            debug_info,
        )
        return True
