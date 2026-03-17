set(LLVM_TARGET_DEFINITIONS TritonGPUDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ttg)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=ttg)

set(LLVM_TARGET_DEFINITIONS TritonGPUTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=ttg)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=ttg)
add_public_tablegen_target(TritonGPUDialectIncGen)

set(LLVM_TARGET_DEFINITIONS CTAEncodingAttr.td)
mlir_tablegen(CTAEncodingAttr.h.inc -gen-attrdef-decls)
add_public_tablegen_target(TritonGPUCTAAttrIncGen)

set(LLVM_TARGET_DEFINITIONS TritonGPUAttrDefs.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(AttrDefs.h.inc -gen-attrdef-decls)

set(LLVM_TARGET_DEFINITIONS TritonGPUAttrImpls.td)
mlir_tablegen(AttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(TritonGPUAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS TritonGPUTypeInterfaces.td)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(TritonGPUTypeInterfacesIncGen)

add_library(TritonGPUIR
  Dialect.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  LinearLayoutConversions.cpp
  Types.cpp
)
add_dependencies(TritonGPUIR
  TritonIR
  TritonGPUDialectIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRLinalgTransforms
  MLIRGPUTransforms
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRTableGen
  clang_rt.ubsan_standalone-x86_64
)
