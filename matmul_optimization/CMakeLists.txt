cmake_minimum_required(VERSION 3.22)
project(matmul_optimization CUDA CXX)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 87)

# add debug symbols to device code
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# Suppress specific CUDA warnings
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe \"--diag_suppress=20054-D\" -Xcudafe \"--diag_suppress=177\" -Xcudafe \"--diag_suppress=550\"")

# configure header file paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY} CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu)  
set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY} CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY} CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})