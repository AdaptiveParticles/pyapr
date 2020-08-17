cmake_minimum_required(VERSION 3.9)
project(PyLibAPR DESCRIPTION "Python wrappers for LibAPR")

set(CMAKE_CXX_STANDARD 14)


# APR build options:
option(PYAPR_USE_OPENMP "Use OpenMP for PyAPR functionality" ON)
option(PYAPR_USE_CUDA "Use CUDA for PyAPR functionality" OFF)

##------------- Required libraries -------------##
find_package(HDF5 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TIFF REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -DHAVE_LIBTIFF")
set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -DHAVE_LIBTIFF")

###############################################################################
# Build LibAPR
###############################################################################
message(STATUS "PyAPR: Building LibAPR")

set(APR_TESTS OFF CACHE BOOL "" FORCE)
set(APR_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(APR_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(APR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(APR_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "" FORCE)
set(APR_USE_LIBTIFF ON CACHE BOOL "" FORCE)
set(APR_USE_OPENMP ${PYAPR_USE_OPENMP} CACHE BOOL "" FORCE)
set(APR_USE_CUDA ${PYAPR_USE_CUDA} CACHE BOOL "" FORCE)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/LibAPR ${HDF5_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})

add_subdirectory("LibAPR")
include_directories(LibAPR/src)
include_directories("LibAPR/external/glm")

###############################################################################
# Handle OpenMP
###############################################################################
find_package(OpenMP)
if(NOT OPENMP_FOUND OR NOT PYAPR_USE_OPENMP)
    message(WARNING "OpenMP support not found or disabled with current compiler. While PyAPR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPYAPR_HAVE_OPENMP -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYAPR_HAVE_OPENMP -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()


###############################################################################
# Handle CUDA
###############################################################################
if(PYAPR_USE_CUDA)
    message(STATUS "APR: Building CUDA for PYAPR")
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread -lineinfo -Xptxas -O3,-v --use_fast_math -DPYAPR_USE_CUDA -DAPR_USE_CUDA")

    enable_language(CUDA)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPYAPR_USE_CUDA -DAPR_USE_CUDA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYAPR_USE_CUDA -DAPR_USE_CUDA")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

endif()

###############################################################################
# Configure compiler options
###############################################################################
# If you ever want to compile with Intel's icc (or any other compiler) provide
# compiler names/paths in cmake command like this:
# CC="icc" CXX="icc" CXXFLAGS="-O3" cmake -DAPR_TESTS=1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic ")
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O4 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic")
    if(NOT WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl -lz")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lz")
endif()

###############################################################################
# Python wrappers
###############################################################################
message(STATUS "PyAPR: Building PYTHON wrappers")

set(PYTHON_VERSION 3)
find_package( PythonInterp ${PYTHON_VERSION} REQUIRED )
find_package( PythonLibs ${PYTHON_VERSION} REQUIRED )

add_subdirectory("pybind11")
add_library(maxflow SHARED
            external/maxflow-v3.04.src/graph.cpp
            external/maxflow-v3.04.src/maxflow.cpp)

#set(MAXFLOW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/maxflow-v3.04.src)

include_directories("pyapr")
include_directories("external")

set(APR_PYTHON_MODULE_NAME _pyaprwrapper)
add_definitions(-DAPR_PYTHON_MODULE_NAME=${APR_PYTHON_MODULE_NAME})
pybind11_add_module(${APR_PYTHON_MODULE_NAME} MODULE wrappers/pythonBind.cpp)
target_include_directories(${APR_PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/wrappers ${CMAKE_CURRENT_SOURCE_DIR}/pyapr ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(${APR_PYTHON_MODULE_NAME} PRIVATE staticLib pybind11::module ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${BLOSC_LIBRARIES} ${ZLIB_LIBRARIES} maxflow)
set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES OUTPUT_NAME ${APR_PYTHON_MODULE_NAME})
set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")