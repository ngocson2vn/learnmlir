cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -std=c++14")
add_link_options("-fuse-ld=lld")

cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
# message(STATUS "PARENT_DIR: ${PARENT_DIR}")
# cmake_path(GET PARENT_DIR PARENT_PATH LEARNMLIR_DIR)
# message(STATUS "LEARNMLIR_DIR: ${LEARNMLIR_DIR}")

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
include_directories(${CMAKE_BINARY_DIR})

# set(LLVM_BINARY_DIR "third_party/llvm-project/build")
# add_subdirectory(third_party/llvm-project/llvm ${LLVM_BINARY_DIR})
# add_subdirectory(third_party/llvm-project/llvm)

find_package(MLIR REQUIRED CONFIG)

# LLVM and MLIR headers
include_directories(third_party/llvm-project/llvm/include)
include_directories(third_party/llvm-project/mlir/include)
set(MLIR_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/llvm-project/mlir/include)

# Some headers are generated automatically, which is why we need to include the following directories
# For llvm/Config/llvm-config.h
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

# include_directories(${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include)
# include_directories(${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include)

set(MLIR_CMAKE_DIR "${PROJECT_SOURCE_DIR}/third_party/llvm-project/mlir/cmake/modules")
set(LLVM_CMAKE_DIR "${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/cmake/modules")

message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR})

set(STABLEHLO_BUILD_EMBEDDED "ON")
add_subdirectory(third_party/mlir-hlo)
include_directories(third_party/mlir-hlo)
include_directories(${CMAKE_BINARY_DIR}/third_party/mlir-hlo)

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
