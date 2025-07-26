cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(llvm-project/llvm)

include_directories(llvm-project/llvm/include)
include_directories(llvm-project/mlir/include)
include_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/include)
include_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/tools/mlir/include)


# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS toy_ops.td)
mlir_tablegen(toy_ops.h.inc -gen-op-decls)
mlir_tablegen(toy_ops.cpp.inc -gen-op-defs)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(toy_dialect 
  toy_dialect.cpp 
  ${CMAKE_CURRENT_BINARY_DIR}/toy_ops.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/toy_ops.cpp.inc
)
add_dependencies(toy_dialect MLIRTableGen MLIRIR MLIRDialect)
target_compile_options(toy_dialect PRIVATE -fno-rtti -g -O0)

add_executable(toy_lowering 
  main.cpp
)
target_compile_options(toy_lowering PRIVATE -fno-rtti -g -O0)

target_link_libraries(toy_lowering PRIVATE
  toy_dialect
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRParser
  MLIRSupport
  MLIRTableGen
  clang_rt.ubsan_standalone-x86_64
)
