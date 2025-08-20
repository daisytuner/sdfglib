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
    $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

# Don't link against PAPI!
# Just link against the dynamic linker
target_link_libraries(daisy_rtl PRIVATE dl arg_capture_io)

set_target_properties(daisy_rtl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS daisy_rtl
    EXPORT sdfglibTargets
    LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT runtime
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT runtime
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT runtime
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)

install(DIRECTORY include/daisy_rtl DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT runtime)
