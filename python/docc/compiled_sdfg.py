import ctypes
from ._docc import Scalar, Array, Pointer, Structure, PrimitiveType

try:
    import numpy as np
except ImportError:
    np = None

_CTYPES_MAP = {
    PrimitiveType.Bool: ctypes.c_bool,
    PrimitiveType.Int8: ctypes.c_int8,
    PrimitiveType.Int16: ctypes.c_int16,
    PrimitiveType.Int32: ctypes.c_int32,
    PrimitiveType.Int64: ctypes.c_int64,
    PrimitiveType.UInt8: ctypes.c_uint8,
    PrimitiveType.UInt16: ctypes.c_uint16,
    PrimitiveType.UInt32: ctypes.c_uint32,
    PrimitiveType.UInt64: ctypes.c_uint64,
    PrimitiveType.Float: ctypes.c_float,
    PrimitiveType.Double: ctypes.c_double,
}


class CompiledSDFG:
    def __init__(self, lib_path, sdfg, shape_sources=None, structure_member_info=None):
        self.lib_path = lib_path
        self.sdfg = sdfg
        self.shape_sources = shape_sources or []
        self.structure_member_info = structure_member_info or {}
        self.lib = ctypes.CDLL(lib_path)
        self.func = getattr(self.lib, sdfg.name)

        # Cache for ctypes structure definitions
        self._ctypes_structures = {}

        # Set up argument types
        self.arg_types = []
        self.arg_docc_types = []  # Keep track of original docc types
        for arg_name in sdfg.arguments:
            arg_type = sdfg.type(arg_name)
            self.arg_docc_types.append(arg_type)
            ct_type = self._get_ctypes_type(arg_type)
            self.arg_types.append(ct_type)

        self.func.argtypes = self.arg_types

        # Set up return type
        self.func.restype = self._get_ctypes_type(sdfg.return_type)

    def _create_ctypes_structure(self, struct_name):
        """Create a ctypes Structure class for the given structure name."""
        if struct_name in self._ctypes_structures:
            return self._ctypes_structures[struct_name]

        if struct_name not in self.structure_member_info:
            raise ValueError(f"Structure '{struct_name}' not found in member info")

        # Get member info: {member_name: (index, type)}
        members = self.structure_member_info[struct_name]
        # Sort by index to get correct order
        sorted_members = sorted(members.items(), key=lambda x: x[1][0])

        # Build _fields_ for ctypes.Structure
        fields = []
        for member_name, (index, member_type) in sorted_members:
            ct_type = self._get_ctypes_type(member_type)
            fields.append((member_name, ct_type))

        # Create the ctypes Structure class dynamically
        class CStructure(ctypes.Structure):
            _fields_ = fields

        self._ctypes_structures[struct_name] = CStructure
        return CStructure

    def _get_ctypes_type(self, docc_type):
        if isinstance(docc_type, Scalar):
            return _CTYPES_MAP.get(docc_type.primitive_type, ctypes.c_void_p)
        elif isinstance(docc_type, Array):
            # Arrays are passed as pointers
            elem_type = _CTYPES_MAP.get(docc_type.primitive_type, ctypes.c_void_p)
            return ctypes.POINTER(elem_type)
        elif isinstance(docc_type, Pointer):
            # Check if pointee is a Structure
            # Note: has_pointee_type() is guaranteed to exist on Pointer instances from C++ bindings
            if docc_type.has_pointee_type():
                pointee = docc_type.pointee_type
                if isinstance(pointee, Structure):
                    # Create ctypes structure and return pointer to it
                    struct_class = self._create_ctypes_structure(pointee.name)
                    return ctypes.POINTER(struct_class)
                elif isinstance(pointee, Scalar):
                    elem_type = _CTYPES_MAP.get(pointee.primitive_type, ctypes.c_void_p)
                    return ctypes.POINTER(elem_type)
            return ctypes.c_void_p
        return ctypes.c_void_p

    def __call__(self, *args):
        # Expand arguments (handle numpy arrays and their shapes)
        expanded_args = list(args)

        # Append unified shape arguments
        for arg_idx, dim_idx in self.shape_sources:
            arg = args[arg_idx]
            if np is not None and isinstance(arg, np.ndarray):
                expanded_args.append(arg.shape[dim_idx])
            else:
                raise ValueError(
                    f"Expected ndarray at index {arg_idx} for shape source"
                )

        if len(expanded_args) != len(self.arg_types):
            raise ValueError(
                f"Expected {len(self.arg_types)} arguments (including implicit shapes), got {len(expanded_args)}"
            )

        converted_args = []
        # Keep references to ctypes structures to prevent garbage collection
        structure_refs = []

        for i, arg in enumerate(expanded_args):
            target_type = self.arg_types[i]
            docc_type = self.arg_docc_types[i] if i < len(self.arg_docc_types) else None

            # Handle numpy arrays
            if np is not None and isinstance(arg, np.ndarray):
                # Check if it's a pointer type
                if hasattr(target_type, "contents"):  # It's a pointer
                    converted_args.append(arg.ctypes.data_as(target_type))
                else:
                    converted_args.append(arg)
            # Handle class instances (structures)
            # Note: has_pointee_type() is guaranteed on Pointer instances
            elif (
                docc_type
                and isinstance(docc_type, Pointer)
                and docc_type.has_pointee_type()
                and isinstance(docc_type.pointee_type, Structure)
            ):
                # Convert Python object to ctypes structure
                struct_name = docc_type.pointee_type.name
                struct_class = self._ctypes_structures.get(struct_name)

                # This should not happen if type setup was done correctly
                if struct_class is None:
                    raise RuntimeError(
                        f"Internal error: Structure '{struct_name}' was not created during type setup. "
                        f"This indicates a bug in the compilation process."
                    )

                # Validate the Python object has the required structure
                if not hasattr(arg, "__dict__"):
                    raise TypeError(
                        f"Expected object with attributes for structure '{struct_name}', "
                        f"but got {type(arg).__name__} without __dict__"
                    )

                # Get member info to know the order
                members = self.structure_member_info[struct_name]
                sorted_members = sorted(members.items(), key=lambda x: x[1][0])

                # Create ctypes structure instance with values from Python object
                struct_values = {}
                for member_name, (index, member_type) in sorted_members:
                    if hasattr(arg, member_name):
                        struct_values[member_name] = getattr(arg, member_name)
                    else:
                        raise ValueError(
                            f"Python object missing attribute '{member_name}' for structure '{struct_name}'"
                        )

                c_struct = struct_class(**struct_values)
                structure_refs.append(c_struct)  # Keep alive
                # Pass pointer to the structure
                converted_args.append(ctypes.pointer(c_struct))
            else:
                converted_args.append(arg)

        return self.func(*converted_args)

    def get_return_shape(self, *args):
        shape_str = self.sdfg.metadata("return_shape")
        if not shape_str:
            return None

        shape_exprs = shape_str.split(",")

        # We need to evaluate these expressions
        # They might contain _s0, _s1 etc.
        # We have shape_sources which maps (arg_idx, dim_idx) -> unique_shape_idx

        # Reconstruct shape values
        shape_values = {}
        for i, (arg_idx, dim_idx) in enumerate(self.shape_sources):
            arg = args[arg_idx]
            if np is not None and isinstance(arg, np.ndarray):
                val = arg.shape[dim_idx]
                shape_values[f"_s{i}"] = val

        # Add scalar arguments to shape_values
        # We assume the first len(args) arguments in sdfg.arguments correspond to the user arguments
        if hasattr(self.sdfg, "arguments"):
            for arg_name, arg_val in zip(self.sdfg.arguments, args):
                if isinstance(arg_val, (int, np.integer)):
                    shape_values[arg_name] = int(arg_val)

        evaluated_shape = []
        for expr in shape_exprs:
            # Simple evaluation using eval with shape_values
            # Warning: eval is unsafe, but here expressions come from our compiler
            try:
                val = eval(expr, {}, shape_values)
                evaluated_shape.append(int(val))
            except Exception:
                return None

        return tuple(evaluated_shape)
