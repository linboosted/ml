cmake_minimum_required(VERSION 3.10)
project(mlpack_example)

set(CMAKE_CXX_STANDARD 17)

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Use pkg-config to find mlpack
pkg_check_modules(MLPACK REQUIRED mlpack)

# Include directories from pkg-config
include_directories(${MLPACK_INCLUDE_DIRS})

# Link directories from pkg-config
link_directories(${MLPACK_LIBRARY_DIRS})

# Add the executable
add_executable(example example.cpp)

# Link libraries from pkg-config
target_link_libraries(example ${MLPACK_LIBRARIES})
