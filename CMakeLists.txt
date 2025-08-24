cmake_minimum_required(VERSION 3.20)
project(ProjectWorkParallelProgramming LANGUAGES CXX)

# Standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# It is recommended with the recent pybind11.
set(PYBIND11_FINDPYTHON NEW)

# Download the stable version of pybind11.
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Python + NumPy (headers + libs)
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Build the Python module. The output on Windows is gram_omp.pyd.
pybind11_add_module(gram_omp
        scripts/gram_omp.cpp
)

# Include the NumPy headers (pybind11/numpy.h).
target_include_directories(gram_omp PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gram_omp PRIVATE OpenMP::OpenMP_CXX)
    if(MSVC)
        # Enhances OpenMP activation with MSVC, when necessary.
        target_compile_options(gram_omp PRIVATE /openmp)
    endif()
endif()

# useful warnings
if(MSVC)
    target_compile_options(gram_omp PRIVATE /W4 /EHsc)
else()
    target_compile_options(gram_omp PRIVATE -Wall -Wextra -Wpedantic)
endif()

