cmake_minimum_required(VERSION 3.10)
project(ToyLowering)

# LLVM dependencies
find_package(LLVM REQUIRED CONFIG)
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
include(${LLVM_CMAKE_DIR}/TableGen.cmake)

# MLIR dependencies
find_package(MLIR REQUIRED CONFIG)
message(STATUS "MLIR_DIR: ${MLIR_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
include_directories(${MLIR_INCLUDE_DIRS})
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DMLIR_TOY_DIALECT)

# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS toy_ops.td)
mlir_tablegen(toy_ops.h.inc -gen-op-decls)
mlir_tablegen(toy_ops.cpp.inc -gen-op-defs)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(toy_lowering
  toy_dialect.cpp
  main.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/toy_ops.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/toy_ops.cpp.inc
)

target_compile_options(toy_lowering PRIVATE -fno-rtti -g -O0)

message(STATUS "UBSAN_LIBRARY_DIR: $ENV{UBSAN_LIBRARY_DIR}")

# Find compiler-rt library path
# UBSAN_LIBRARY is necessary to resolve the error "undefined reference to __ubsan_handle_alignment_assumption_abort"
find_library(UBSAN_LIBRARY
  NAMES clang_rt.ubsan_standalone
  PATHS $ENV{UBSAN_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

message(STATUS "UBSAN_LIBRARY: ${UBSAN_LIBRARY}")

target_link_libraries(toy_lowering PRIVATE
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
  ${UBSAN_LIBRARY}
)
