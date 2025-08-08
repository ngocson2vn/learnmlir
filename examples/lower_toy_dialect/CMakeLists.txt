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

# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS toy_ops.td)
mlir_tablegen(toy_dialect.h.inc -gen-dialect-decls)
mlir_tablegen(toy_dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(toy_ops.h.inc -gen-op-decls)
mlir_tablegen(toy_ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpsIncGen)

add_library(toy_dialect 
  toy_dialect.cpp 
)
add_dependencies(toy_dialect ToyOpsIncGen)
target_compile_options(toy_dialect PRIVATE -fno-rtti -g -O0)

# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS toy_passes.td)
mlir_tablegen(toy_passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ToyConversionPassIncGen)

add_library(toy_passes 
  toy_passes.cpp
  # mock.cpp
)
add_dependencies(toy_passes ToyConversionPassIncGen)
target_compile_options(toy_passes PRIVATE -fno-rtti -g -O0)

add_executable(main 
  main.cpp
)
target_compile_options(main PRIVATE -fno-rtti -g -O0)
target_link_libraries(main PRIVATE
  toy_dialect
  toy_passes
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRParser
  MLIRSupport
  MLIRTableGen
  clang_rt.ubsan_standalone-x86_64
)
