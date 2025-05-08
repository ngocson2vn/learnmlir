#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "toy_ops.h.inc"

namespace mlir {
namespace toy {

class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "toy"; }
};

} // namespace toy
} // namespace mlir

#endif // TOY_DIALECT_H
