# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options)

# tutorials

add_executable(matmul_tuner
    matmul_replayer.cpp
)
target_link_libraries(matmul_tuner PRIVATE sdfg::sdfg sdfg::rpc Boost::program_options)
target_include_directories(matmul_tuner PRIVATE $<TARGET_PROPERTY:sdfglib,INTERFACE_INCLUDE_DIRECTORIES>)
