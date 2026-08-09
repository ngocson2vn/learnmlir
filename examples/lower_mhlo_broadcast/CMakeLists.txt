cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -std=c++14 -Wno-deprecated-declarations")

# Use ld.lld linker
add_link_options("-fuse-ld=lld")

#============================================================================
# MLIR and LLVM
#============================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# LLVM and MLIR headers
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")


#============================================================================
# MHLO
#============================================================================
set(STABLEHLO_BUILD_EMBEDDED "ON")
add_subdirectory(third_party/mlir-hlo)
include_directories(third_party/mlir-hlo)
include_directories(${CMAKE_BINARY_DIR}/third_party/mlir-hlo)


#============================================================================
# main
#============================================================================
cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR})
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
include_directories(${CMAKE_BINARY_DIR})
add_executable(main 
  main.cpp
)
target_compile_options(main PRIVATE -g -O0)

set(CLANG_LIB_DIR ${LLVM_BINARY_DIR}/lib/clang/22/lib/x86_64-unknown-linux-gnu)
message(STATUS "CLANG_LIB_DIR: ${LLVM_BINARY_DIR}/lib/clang/22/lib/x86_64-unknown-linux-gnu")

target_link_libraries(main PRIVATE
  MhloRegisterDialects
  AllMhloPasses
  StablehloPasses
  MLIRBufferTransforms

  # MLIR
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

  ${CLANG_LIB_DIR}/libclang_rt.ubsan_standalone.a
)
