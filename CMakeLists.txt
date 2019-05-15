cmake_minimum_required(VERSION 3.9)
project(PyLibAPR DESCRIPTION "Python wrappers for LibAPR")

set(CMAKE_CXX_STANDARD 14)


# APR build options:
option(PYAPR_USE_OPENMP "Use OpenMP for PyAPR functionality" ON)

##------------- Required libraries -------------##
find_package(HDF5 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TIFF REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_CFLAGS} -DHAVE_LIBTIFF")
set(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -DHAVE_LIBTIFF")

##------------- LibAPR -------------##
message(STATUS "PyAPR: Building LibAPR")

set(APR_TESTS OFF CACHE BOOL "" FORCE)
set(APR_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(APR_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(APR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(APR_USE_CUDA OFF CACHE BOOL "" FORCE)
set(APR_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "" FORCE)
set(APR_USE_LIBTIFF ON CACHE BOOL "" FORCE)
set(APR_USE_OPENMP ON CACHE BOOL "" FORCE)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/LibAPR ${HDF5_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})

add_subdirectory("LibAPR")
include_directories(LibAPR/src)


##------------- Handle OpenMP -------------##
find_package(OpenMP)
if(NOT OPENMP_FOUND OR NOT PYAPR_USE_OPENMP)
    message(WARNING "OpenMP support not found or disabled with current compiler. While PyAPR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPYAPR_HAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYAPR_HAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()

##------------- Python wrappers -------------##
message(STATUS "PyAPR: Building PYTHON wrappers")

add_subdirectory("pybind11")

set(APR_PYTHON_MODULE_NAME _pyaprwrapper)
add_definitions(-DAPR_PYTHON_MODULE_NAME=${APR_PYTHON_MODULE_NAME})
add_library(${APR_PYTHON_MODULE_NAME} MODULE wrappers/pythonBind.cpp)
target_include_directories(${APR_PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/wrappers)
target_link_libraries(${APR_PYTHON_MODULE_NAME} PRIVATE staticLib pybind11::module ${HDF5_LIBRARIES} ${TIFF_LIBRARIES} ${BLOSC_LIBRARIES} ${ZLIB_LIBRARIES})
set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES OUTPUT_NAME ${APR_PYTHON_MODULE_NAME})
set_target_properties(${APR_PYTHON_MODULE_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")