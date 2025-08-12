include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(daisy_rtl STATIC
    src/capture.cpp
    src/instrumentation.cpp
)

target_include_directories(daisy_rtl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Don't link against PAPI!
# Just link against the dynamic linker
target_link_libraries(daisy_rtl PRIVATE dl sdfglib::arg_capture_io)

set_target_properties(daisy_rtl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS daisy_rtl
    LIBRARY DESTINATION lib/docc-${DOCC_VERSION}/lib COMPONENT runtime
    ARCHIVE DESTINATION lib/docc-${DOCC_VERSION}/lib COMPONENT runtime
    PUBLIC_HEADER DESTINATION lib/docc-${DOCC_VERSION}/include COMPONENT runtime
    INCLUDES DESTINATION lib/docc-${DOCC_VERSION}/include COMPONENT runtime
)

install(DIRECTORY include/daisy_rtl DESTINATION lib/docc-${DOCC_VERSION}/include COMPONENT runtime)
