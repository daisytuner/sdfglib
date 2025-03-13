include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Original Library
add_library(daisy_rtl STATIC
    src/daisy_instrumentation.cpp
)

target_include_directories(daisy_rtl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(daisy_rtl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS daisy_rtl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
