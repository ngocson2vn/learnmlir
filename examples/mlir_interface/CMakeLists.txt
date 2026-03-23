cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

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

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR})

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Generate Interface files
set(LLVM_TARGET_DEFINITIONS ExampleInterface.td)
mlir_tablegen(ExampleInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ExampleInterface.cpp.inc -gen-op-interface-defs)
add_custom_target(ExampleInterfaceIncGen DEPENDS ExampleInterface.h.inc ExampleInterface.cpp.inc)

# Generate Dialect AND Operations files from ExampleDialect.td
set(LLVM_TARGET_DEFINITIONS ExampleDialect.td)
mlir_tablegen(ExampleDialect.h.inc -gen-dialect-decls -dialect=example)
mlir_tablegen(ExampleDialect.cpp.inc -gen-dialect-defs -dialect=example)
mlir_tablegen(ExampleOps.h.inc -gen-op-decls)
mlir_tablegen(ExampleOps.cpp.inc -gen-op-defs)
add_custom_target(ExampleDialectIncGen DEPENDS 
  ExampleDialect.h.inc ExampleDialect.cpp.inc 
  ExampleOps.h.inc ExampleOps.cpp.inc
)

add_library(ExampleIR
  Interface.cpp
  Dialect.cpp
)
add_dependencies(ExampleIR
  ExampleInterfaceIncGen
  ExampleDialectIncGen
)

add_executable(main 
  main.cpp
)
add_dependencies(main
  ExampleIR
)
target_compile_options(main PRIVATE -g -O0)
target_link_libraries(main PRIVATE
  ExampleIR
  MLIRDialect
  MLIRRegisterAllDialects
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRParser
  MLIRSupport
  MLIRTableGen

  # LLVM
  LLVMTableGen
  LLVMSupport
  LLVMTarget
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Disassembler
  LLVMX86TargetMCA
  LLVMX86Desc
  LLVMX86Info

  clang_rt.ubsan_standalone-x86_64
)
