// llvm-project/mlir/include/mlir/Support/InterfaceSupport.h
namespace mlir {
namespace detail {

template<>
class Interface<lmhlo::LmhloOp, Operation*, detail::LmhloOpInterfaceTraits, Op<lmhlo::LmhloOp>, OpTrait::TraitBase> : public Op<lmhlo::LmhloOp> {
public:
  using Concept = typename detail::LmhloOpInterfaceTraits::Concept;
  
  template <typename T>
  using Model = typename detail::LmhloOpInterfaceTraits::template Model<T>;
  
  template <typename T>
  using FallbackModel = typename detail::LmhloOpInterfaceTraits::template FallbackModel<T>;

  using InterfaceBase = Interface<lmhlo::LmhloOp, Operation*, detail::LmhloOpInterfaceTraits, Op<lmhlo::LmhloOp>, OpTrait::TraitBase>;
  
  template <typename T, typename U>
  using ExternalModel = typename detail::LmhloOpInterfaceTraits::template ExternalModel<T, U>;
  
  using ValueType = Operation*;

  /// This is a special trait that registers a given interface with an object.
  template <typename ConcreteT>
  struct Trait : public OpTrait::TraitBase<ConcreteT, Trait> {
    using ModelT = Model<ConcreteT>;

    /// Define an accessor for the ID of this interface.
    static TypeID getInterfaceID() { return TypeID::get<lmhlo::LmhloOp>(); }
  };

  /// Construct an interface from an instance of the value type.
  explicit Interface(Operation* t = Operation*())
      : Op<lmhlo::LmhloOp>(t),
        conceptImpl(t ? lmhlo::LmhloOp::getInterfaceFor(t) : nullptr) {
    assert((!t || conceptImpl) &&
           "expected value to provide interface instance");
  }
  Interface(std::nullptr_t) : Op<lmhlo::LmhloOp>(Operation*()), conceptImpl(nullptr) {}

  /// Construct an interface instance from a type that implements this
  /// interface's trait.
  template <typename T,
            std::enable_if_t<std::is_base_of<Trait<T>, T>::value> * = nullptr>
  Interface(T t)
      : Op<lmhlo::LmhloOp>(t),
        conceptImpl(t ? lmhlo::LmhloOp::getInterfaceFor(t) : nullptr) {
    assert((!t || conceptImpl) &&
           "expected value to provide interface instance");
  }

  /// Constructor for a known concept.
  Interface(Operation* t, const Concept *conceptImpl)
      : Op<lmhlo::LmhloOp>(t), conceptImpl(const_cast<Concept *>(conceptImpl)) {
    assert(!t || lmhlo::LmhloOp::getInterfaceFor(t) == conceptImpl);
  }

  /// Constructor for DenseMapInfo's empty key and tombstone key.
  Interface(Operation* t, std::nullptr_t) : Op<lmhlo::LmhloOp>(t), conceptImpl(nullptr) {}

  /// Support 'classof' by checking if the given object defines the concrete
  /// interface.
  static bool classof(Operation* t) { return lmhlo::LmhloOp::getInterfaceFor(t); }

  /// Define an accessor for the ID of this interface.
  static TypeID getInterfaceID() { return TypeID::get<lmhlo::LmhloOp>(); }

protected:
  /// Get the raw concept in the correct derived concept type.
  const Concept *getImpl() const { return conceptImpl; }
  Concept *getImpl() { return conceptImpl; }

private:
  /// A pointer to the impl concept object.
  Concept *conceptImpl;
};

} // namespace detail
} // namespace mlir

namespace mlir {

template <>
class OpInterface<lmhlo::LmhloOp, detail::LmhloOpInterfaceTraits>
  : public detail::Interface<lmhlo::LmhloOp, Operation *, detail::LmhloOpInterfaceTraits, Op<lmhlo::LmhloOp>, OpTrait::TraitBase> {
public:
  using Base = OpInterface<lmhlo::LmhloOp, detail::LmhloOpInterfaceTraits>;
  using InterfaceBase = detail::Interface<lmhlo::LmhloOp, Operation *, detail::LmhloOpInterfaceTraits, Op<lmhlo::LmhloOp>, OpTrait::TraitBase>;

  /// Inherit the base class constructor.
  using InterfaceBase::InterfaceBase;

protected:
  /// Returns the impl interface instance for the given operation.
  static typename InterfaceBase::Concept* getInterfaceFor(Operation* op) {
    OperationName name = op->getName();

    // Access the raw interface from the operation info.
    if (std::optional<RegisteredOperationName> rInfo =
            name.getRegisteredInfo()) {
      if (auto *opIface = rInfo->getInterface<lmhlo::LmhloOp>())
        return opIface;
      // Fallback to the dialect to provide it with a chance to implement this
      // interface for this operation.
      return rInfo->getDialect().getRegisteredInterfaceForOp<lmhlo::LmhloOp>(
          op->getName());
    }
    // Fallback to the dialect to provide it with a chance to implement this
    // interface for this operation.
    if (Dialect *dialect = name.getDialect())
      return dialect->getRegisteredInterfaceForOp<lmhlo::LmhloOp>(name);
    return nullptr;
  }

  /// Allow access to `getInterfaceFor`.
  friend InterfaceBase;
};

} // namespace mlir

// llvm-project/mlir/include/mlir/IR/OpDefinition.h
namespace mlir {
namespace lmhlo {
class LmhloOp;
namespace detail {
struct LmhloOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
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
struct LmhloOpTrait : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::Trait<ConcreteOp> {
  /// Return the operand that is the output buffer
  Value getResultBuffer() {
    /// By default, the result buffer is the last operand
    unsigned num_operands = this->getOperation()->getNumOperands();
    if (num_operands >= 1) {
      return this->getOperation()->getOperand(num_operands - 1);
    }
    return nullptr;
  }
};

} // namespace detail

class LmhloOp : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LmhloOpTrait<ConcreteOp> {};
  /// Return the operand that is the output buffer
  Value getResultBuffer();
};

} // namespace lmhlo
} // namespace mlir