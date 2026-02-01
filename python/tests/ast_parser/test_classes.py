from docc.python import native


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_structure_type_mapping():
    from docc.sdfg import Structure, Pointer
    from docc.python import _map_python_type

    # Test that a class is mapped to Pointer(Structure)
    result = _map_python_type(Point)
    assert isinstance(result, Pointer)
    assert result.has_pointee_type()
    assert isinstance(result.pointee_type, Structure)
    assert result.pointee_type.name == "Point"


def test_class_instance_type_inference():
    """Test that class instances are correctly inferred as Pointer to Structure"""
    from docc.sdfg import Structure, Pointer
    from docc.python import PythonProgram

    point = Point(3.0, 4.0)
    program = PythonProgram(lambda p: p, target="none")

    # Test type inference for class instance
    inferred_type = program._infer_type(point)
    assert isinstance(inferred_type, Pointer)
    assert inferred_type.has_pointee_type()
    assert isinstance(inferred_type.pointee_type, Structure)
    assert inferred_type.pointee_type.name == "Point"


def test_class_type_string_representation():
    """Test that Structure types are properly represented in signatures"""
    from docc.sdfg import Structure, Pointer
    from docc.python import PythonProgram

    program = PythonProgram(lambda p: p, target="none")
    point_type = Pointer(Structure("Point"))

    # Test type to string conversion
    type_str = program._type_to_str(point_type)
    assert "Pointer" in type_str
    assert "Structure" in type_str
    assert "Point" in type_str


def test_structure_member_info_generation():
    """Test that structure member info is correctly generated with indices"""
    from docc.python import PythonProgram

    def get_x(p: Point) -> float:
        return p.x

    point = Point(3.0, 4.0)
    program = PythonProgram(get_x, target="none")

    # Build SDFG to trigger structure registration
    arg_types = [program._infer_type(point)]
    sdfg = program._build_sdfg(arg_types, [point], {}, 0)

    # The structure should have been registered during build
    assert sdfg is not None


def test_structure_member_access_first_member():
    """Test accessing the first member (index 0) of a structure"""

    @native
    def get_x(p: Point) -> float:
        return p.x

    point = Point(3.0, 4.0)
    result = get_x(point)
    assert result == 3.0


def test_structure_member_access_second_member():
    """Test accessing the second member (index 1) of a structure"""

    @native
    def get_y(p: Point) -> float:
        return p.y

    point = Point(5.0, 7.0)
    result = get_y(point)
    assert result == 7.0


def test_structure_member_access_third_member():
    """Test accessing the third member (index 2) of a 3D point"""

    @native
    def get_z(p: Point3D) -> float:
        return p.z

    point = Point3D(1.0, 2.0, 3.0)
    result = get_z(point)
    assert result == 3.0


def test_structure_multiple_member_access():
    """Test accessing multiple members in the same function"""

    @native
    def sum_coords(p: Point) -> float:
        return p.x + p.y

    point = Point(3.0, 4.0)
    result = sum_coords(point)
    assert result == 7.0


def test_structure_member_access_with_operations():
    """Test using member access in calculations"""

    @native
    def distance_from_origin(p: Point) -> float:
        return (p.x * p.x + p.y * p.y) ** 0.5

    point = Point(3.0, 4.0)
    result = distance_from_origin(point)
    assert abs(result - 5.0) < 1e-10
