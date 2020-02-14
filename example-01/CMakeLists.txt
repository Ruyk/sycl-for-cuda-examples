cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

add_executable(vector_addition vector_addition.cu)

target_compile_features(vector_addition PUBLIC cxx_std_11)

set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET vector_addition PROPERTY BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")


if (NOT SYCL_ROOT) 
  message(FATAL_ERROR "No SYCL installation detected")
endif(NOT SYCL_ROOT)


set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/11.0.0/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")

add_executable (sycl_vector_addition vector_addition.cpp)

target_compile_features(sycl_vector_addition PUBLIC cxx_std_17)
target_compile_options(sycl_vector_addition PUBLIC "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice,spir64-unknown-linux-sycldevice")
target_link_libraries(sycl_vector_addition PUBLIC "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice,spir64-unknown-linux-sycldevice")

target_include_directories(sycl_vector_addition PUBLIC ${SYCL_INCLUDE_DIR})
target_link_libraries(sycl_vector_addition PUBLIC ${SYCL_LIB})

# set_property(TARGET sycl_vector_addition PROPERTY BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
