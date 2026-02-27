set(LLVM_TARGET_DEFINITIONS TritonGPUDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ttg)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=ttg)
add_public_tablegen_target(DialectsIncGen)

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
