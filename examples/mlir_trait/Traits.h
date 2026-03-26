#pragma once

#include "mlir/Dialect/Traits.h"

namespace mlir {
namespace example {

// The template parameter 'ConcreteType' will be 'CustomOp' when instantiated
template <typename ConcreteType>
class VerifyMagicNumberTrait : public OpTrait::TraitBase<ConcreteType, VerifyMagicNumberTrait> {
public:
  // MLIR will automatically call this static method during operation verification
  static LogicalResult verifyTrait(Operation *op) {
    int magicNumber = -1;

    // Look up an attribute by string name, completely agnostic to the specific Op class or Interfaces
    auto attr = op->getAttrOfType<IntegerAttr>("magic_number");
    if (attr) {
      magicNumber = attr.getInt();
    }

    // 2. Enforce your safety rule
    // For example, let's enforce that the magic number cannot be negative
    if (magicNumber < 0) {
        return op->emitOpError() << "safety verification failed: magic number cannot be negative!";
    }

    // If all checks pass, return success
    llvm::outs() << "VerifyMagicNumberTrait::verifyTrait() success for op=" << op->getName() << "\n";
    return success();
  }

  int64_t getMagicNumber() {
    int64_t magicNumber = -1;

    // Look up an attribute by string name, completely agnostic to the specific Op class or Interfaces
    Operation* op = static_cast<ConcreteType*>(this)->getOperation();
    auto attr = op->getAttrOfType<IntegerAttr>("magic_number");
    if (attr) {
      magicNumber = attr.getInt();
    }

    return magicNumber;
  }

  // A custom utility method, NOT verification
  bool isMagicNumberEven() {
    return getMagicNumber() % 2 == 0;
  }
};

} // namespace example
} // namespace mlir