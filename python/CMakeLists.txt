cmake_minimum_required(VERSION 3.26)

# Python dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

find_package(nlohmann_json REQUIRED)

# Define module
set(PYTHON_SOURCES
    bindings/bindings.cpp
    bindings/py_structured_sdfg.cpp
    bindings/py_structured_sdfg_builder.cpp
    bindings/types/py_types.cpp
)

pybind11_add_module(sdfg-py ${PYTHON_SOURCES})
set_target_properties(sdfg-py PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_sdfg"
)

target_link_libraries(sdfg-py PRIVATE sdfg::sdfg sdfg::opt sdfg::rpc nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})

# Install artifacts for the Python wheel
install(TARGETS sdfg-py DESTINATION docc)

install(FILES
    $<TARGET_FILE:daisy_rtl>
    $<TARGET_FILE:arg_capture_io>
    DESTINATION docc
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../rtl/include/"
    DESTINATION docc/include
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../arg-capture-io/include/"
    DESTINATION docc/include
)
