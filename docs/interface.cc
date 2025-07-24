#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/OpDefinition.h"
//=====================================================================================================================
// Op template
// llvm-project/mlir/include/mlir/IR/OpDefinition.h
//=====================================================================================================================
namespace mlir {

/// This provides public APIs that all operations should have.  The template
/// argument 'ConcreteType' should be the concrete type by CRTP and the others
/// are base classes by the policy pattern.
template <typename ConcreteType, template <typename T> class... Traits>
class Op : public OpState, public Traits<ConcreteType>... {
  // Omit for brevity
};

} // namespace mlir


//=====================================================================================================================
// MLIR-HLO
// org_tensorflow/tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h.inc
//=====================================================================================================================
#include "mlir-hlo/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/lhlo/IR/lhlo_ops.h"
namespace mlir {
namespace lmhlo {

class AddOp : public ::mlir::Op<AddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::lmhlo::LmhloOp::Trait, ::mlir::OpTrait::SameTypeOperands, ::mlir::OpTrait::Elementwise> {
  // omit for brevity
};

} // namespace lmhlo
} // namespace mlir


// Inheritance: AddOp -> Op -> LmhloOp::Trait


// Step 0: The main program create a MLIRContext context
mlir::DialectRegistry registry;
registry.insert<mlir::func::FuncDialect>();
registry.insert<mlir::mhlo::MhloDialect>();
registry.insert<mlir::lmhlo::LmhloDialect>();
mlir::registerAllDialects(registry);
MLIRContext context(registry);

// Step 1: context loads LmhloDialect dialect by creating a LmhloDialect object
context.loadAllAvailableDialects();

// Step 2: LmhloDialect constructor executes
// org_tensorflow/tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.cc
LmhloDialect::LmhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<LmhloDialect>()) {
  context->loadDialect<mhlo::MhloDialect>();
  addOperations<
#define GET_OP_LIST
#include "lhlo/IR/lhlo_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lhlo/IR/lhlo_ops_structs.cc.inc"
      >();
}

// addOperations<..., AddOp, ...>()
// llvm-project/mlir/include/mlir/IR/Dialect.h
template <typename... Args>
void addOperations() {
  // This initializer_list argument pack expansion is essentially equal to
  // using a fold expression with a comma operator. Clang however, refuses
  // to compile a fold expression with a depth of more than 256 by default.
  // There seem to be no such limitations for initializer_list.
  (void)std::initializer_list<int>{
      0, (RegisteredOperationName::insert<Args>(*this), 0)...};
}


// llvm-project/mlir/include/mlir/IR/OperationSupport.h
/// This is a "type erased" representation of a registered operation. This
/// should only be used by things like the AsmPrinter and other things that need
/// to be parameterized by generic operation hooks. Most user code should use
/// the concrete operation types.
class RegisteredOperationName : public OperationName {
public:
  /// Lookup the registered operation information for the given operation.
  /// Returns None if the operation isn't registered.
  static Optional<RegisteredOperationName> lookup(StringRef name,
                                                  MLIRContext *ctx);

  /// Register a new operation in a Dialect object.
  /// This constructor is used by Dialect objects when they register the list of
  /// operations they contain.
  template <typename T> // T = AddOp
  static void insert(Dialect &dialect) {
    insert(T::getOperationName(), dialect, TypeID::get<T>(),
           T::getParseAssemblyFn(), T::getPrintAssemblyFn(),
           T::getVerifyInvariantsFn(), T::getVerifyRegionInvariantsFn(),
           T::getFoldHookFn(), T::getGetCanonicalizationPatternsFn(),
           T::getInterfaceMap(), T::getHasTraitFn(), T::getAttributeNames(),
           T::getPopulateDefaultAttrsFn());
  }
};

// T = AddOp => T::getInterfaceMap() => AddOp::getInterfaceMap()
// getInterfaceMap() is defined in the Op template
// llvm-project/mlir/include/mlir/IR/OpDefinition.h
static detail::InterfaceMap getInterfaceMap() {
  // Traits is a pack name
  // ConcreteType = AddOp
  return detail::InterfaceMap::template get<Traits<ConcreteType>...>();
  //     => detail::InterfaceMap::template get<..., LmhloOp::Trait<AddOp>, ...>()
}


// InterfaceMap
// llvm-project/mlir/include/mlir/Support/InterfaceSupport.h
/// Construct an InterfaceMap with the given set of template types. For
/// convenience given that object trait lists may contain other non-interface
/// types, not all of the types need to be interfaces. The provided types that
/// do not represent interfaces are not added to the interface map.
template <typename... Types>
static InterfaceMap get() {
  constexpr size_t numInterfaces = num_interface_types_t<Types...>::value; // Refer to ./num_interface_types_t.h
  if constexpr (numInterfaces == 0)
    return InterfaceMap();

  std::array<std::pair<TypeID, void *>, numInterfaces> elements;
  std::pair<TypeID, void *> *elementIt = elements.data();
  (void)elementIt;
  (addModelAndUpdateIterator<Types>(elementIt), ...);
  return InterfaceMap(elements);
}

/// Assign the interface model of the type to the given opaque element
/// iterator and increment it.
template <typename T> // T = LmhloOp::Trait<AddOp>
static inline std::enable_if_t<detect_get_interface_id<T>::value>
addModelAndUpdateIterator(std::pair<TypeID, void *> *&elementIt) {
  *elementIt = {T::getInterfaceID(), new (malloc(sizeof(typename T::ModelT)))
                                          typename T::ModelT()};
  ++elementIt;
}

// Inheritance: LmhloOp::Trait<AddOp> -> detail::LmhloOpTrait<AddOp> -> OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::Trait<AddOp>
// T::getInterfaceID() = LmhloOp::Trait<AddOp>::getInterfaceID() = OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::Trait<AddOp>::getInterfaceID()


// examples/bazel-out/k8-dbg/bin/external/org_tensorflow/tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.h.inc
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace lmhlo {
class LmhloOp;
namespace detail {
struct LmhloOpInterfaceTraits {
  struct Concept {
    Value (*getResultBuffer)(const Concept *impl, ::mlir::Operation *);
  };

  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::lmhlo::LmhloOp;
    Model() : Concept{getResultBuffer} {}

    static inline Value getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };

  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::lmhlo::LmhloOp;
    FallbackModel() : Concept{getResultBuffer} {}

    static inline Value getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };

  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    Value getResultBuffer(::mlir::Operation *tablegen_opaque_val) const;
  };
};

template <typename ConcreteOp>
struct LmhloOpTrait;

} // namespace detail

class LmhloOp : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::OpInterface;

  template <typename ConcreteOp>
  struct Trait : public detail::LmhloOpTrait<ConcreteOp> {};
  /// Return the operand that is the output buffer
  Value getResultBuffer();
};

namespace detail {
  template <typename ConcreteOp>
  struct LmhloOpTrait : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the operand that is the output buffer
    Value getResultBuffer() {
      /// By default, the result buffer is the last operand 
        unsigned num_operands = this->getOperation()->getNumOperands();
        if (num_operands > 1) {
          return this->getOperation()->getOperand(num_operands - 1);
        }
        return nullptr;
    }
  };
}// namespace detail


template<typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::Model<ConcreteOp>::getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultBuffer();
}

template<typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::FallbackModel<ConcreteOp>::getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultBuffer(tablegen_opaque_val);
}

template<typename ConcreteModel, typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getResultBuffer(::mlir::Operation *tablegen_opaque_val) const {
/// By default, the result buffer is the last operand 
        unsigned num_operands = this->getOperation()->getNumOperands();
        if (num_operands > 1) {
          return this->getOperation()->getOperand(num_operands - 1);
        }
        return nullptr;
}

} // namespace lmhlo
} // namespace mlir


// Finally, context stores the LmhloDialect unique ptr into `DenseMap<StringRef, std::unique_ptr<Dialect>> loadedDialects`

// main.cc
auto op = ::llvm::dyn_cast<lmhlo::LmhloOp>(addOp.getOperation());
// This code works because of `OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::Trait<AddOp>::getInterfaceID()`

// Sample code: [Op and OpInterface and Traits](../examples/lower_lmhlo_ops/)