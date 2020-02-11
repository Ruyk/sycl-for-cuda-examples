cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

add_executable(vector_addition vector_addition.cu)

target_compile_features(vector_addition PUBLIC cxx_std_11)

set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET vector_addition PROPERTY BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
