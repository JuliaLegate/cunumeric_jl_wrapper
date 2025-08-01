cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(cuNumericWrapper)

set(cuNumericWrapperVersion 0.0.1)
message(STATUS "Project version: v${cuNumericWrapperVersion}")
set(CXX_CUNUMERICJL_WRAPPER cunumeric_jl_wrapper)
set(C_INTERFACE_LIB cunumeric_c_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(CUDAToolkit 12.2 REQUIRED)
find_package(legate REQUIRED)
find_package(cupynumeric REQUIRED)

# CxxWrap Stuff
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

set(SOURCES
    src/wrapper.cpp
    src/types.cpp
    src/cuda.cpp
)

add_library(${CXX_CUNUMERICJL_WRAPPER} SHARED ${SOURCES})
set_target_properties(${CXX_CUNUMERICJL_WRAPPER} PROPERTIES VERSION ${cuNumericWrapperVersion})

target_link_libraries(${CXX_CUNUMERICJL_WRAPPER} PRIVATE  
                        cupynumeric::cupynumeric
                        legate::legate
                        JlCxx::cxxwrap_julia
                        JlCxx::cxxwrap_julia_stl
                    )
target_include_directories(${CXX_CUNUMERICJL_WRAPPER} PRIVATE include ${CUDAToolkit_INCLUDE_DIRS})
install(TARGETS ${CXX_CUNUMERICJL_WRAPPER} DESTINATION lib)

# C- API
set(C-SOURCES
    src/ndarray.cpp
)

add_library(${C_INTERFACE_LIB} SHARED ${C-SOURCES})

set_target_properties(${C_INTERFACE_LIB} PROPERTIES VERSION ${cuNumericWrapperVersion})

target_link_libraries(${C_INTERFACE_LIB} PRIVATE
                cupynumeric::cupynumeric
                legate::legate
                )
target_include_directories(${C_INTERFACE_LIB} PRIVATE include ${CUDAToolkit_INCLUDE_DIRS})
install(TARGETS ${C_INTERFACE_LIB} DESTINATION lib)