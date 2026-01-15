cmake_minimum_required(VERSION 3.26)

# Python dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

find_package(nlohmann_json REQUIRED)

# Define module
set(PYTHON_SOURCES
    src/bindings.cpp
    src/py_structured_sdfg.cpp
    src/py_structured_sdfg_builder.cpp
    src/types/py_types.cpp
)

pybind11_add_module(docc-py ${PYTHON_SOURCES})
set_target_properties(docc-py PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_docc"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/docc"
)

target_compile_definitions(docc-py PRIVATE
    DOCC_VERSION=\"${SDFGLIB_VERSION}\"
)
target_link_libraries(docc-py PRIVATE sdfglib::sdfglib nlohmann_json::nlohmann_json)

# Install module
install(TARGETS docc-py
    LIBRARY DESTINATION ${Python3_SITELIB}
)
