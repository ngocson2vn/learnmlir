cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17)

cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
message(STATUS "PARENT_DIR: ${PARENT_DIR}")
cmake_path(GET PARENT_DIR PARENT_PATH LEARNMLIR_DIR)
message(STATUS "LEARNMLIR_DIR: ${LEARNMLIR_DIR}")

set(LLVM_BINARY_DIR "${LEARNMLIR_DIR}/llvm-project/build")
add_subdirectory(${LEARNMLIR_DIR}/llvm-project/llvm ${LLVM_BINARY_DIR})

# LLVM and MLIR headers
include_directories(${LEARNMLIR_DIR}/llvm-project/llvm/include)
include_directories(${LEARNMLIR_DIR}/llvm-project/mlir/include)

# Some headers are generated automatically, which is why we need to include the following directories
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(.)

# common
add_subdirectory(common)

# frontend
add_subdirectory(frontend)

# middleend
add_subdirectory(middleend)

# backend
add_subdirectory(backend)

# Find the CUDA runtime library
find_library(CUDA_RUNTIME_LIBRARY
  NAMES cudart
  PATHS ${CUDA_DIR}/lib64
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_RUNTIME_LIBRARY: ${CUDA_RUNTIME_LIBRARY}")

# main
add_executable(main 
  main.cpp
)
target_compile_options(main PRIVATE -fno-rtti -g -O0)
target_link_libraries(main PRIVATE
  frontend
  middleend
  backend
  common
  MLIRIR
  MLIRDialect
  MLIRRegisterAllExtensions
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRNVVMDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRLinalgTransforms
  MLIRSCFToGPU
  MLIRGPUTransforms
  MLIRSCFToControlFlow
  MLIRGPUToNVVMTransforms
  MLIRConvertToLLVMPass
  MLIRBuiltinToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRParser
  MLIRSupport
  MLIRTableGen
  LLVMTableGen
  clang_rt.ubsan_standalone-x86_64
  ${CUDA_RUNTIME_LIBRARY}
)
