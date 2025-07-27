cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17)

cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
message(STATUS "PARENT_DIR: ${PARENT_DIR}")
cmake_path(GET PARENT_DIR PARENT_PATH LEARNMLIR_DIR)
message(STATUS "LEARNMLIR_DIR: ${LEARNMLIR_DIR}")

set(LLVM_BINARY_DIR "${LEARNMLIR_DIR}/llvm-project/build")
add_subdirectory(${LEARNMLIR_DIR}/llvm-project/llvm ${LLVM_BINARY_DIR})

# Generate operation definitions
include_directories(${LEARNMLIR_DIR}/llvm-project/llvm/include)
include_directories(${LEARNMLIR_DIR}/llvm-project/mlir/include)

# Some headers are generated automatically, which is why we need to include the following directories
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)


add_executable(main 
  main.cpp
)
target_compile_options(main PRIVATE -fno-rtti -g -O0)
add_dependencies(main MLIRTableGen MLIRIR MLIRDialect)

target_link_libraries(main PRIVATE
  MLIRIR
  MLIRDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRParser
  MLIRSupport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)
