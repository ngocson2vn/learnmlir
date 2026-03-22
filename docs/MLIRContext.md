# MLIRContext
llvm-project/mlir/include/mlir/IR/MLIRContext.h
```C++
// Forward declaration
class MLIRContextImpl;

class MLIRContext {
public:
  enum class Threading { DISABLED, ENABLED };
  /// Create a new Context.
  explicit MLIRContext(Threading multithreading = Threading::ENABLED);
  explicit MLIRContext(const DialectRegistry &registry,
                       Threading multithreading = Threading::ENABLED);

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() { return *impl; }

private:
  const std::unique_ptr<MLIRContextImpl> impl;
```

llvm-project/mlir/lib/IR/MLIRContext.cpp
```C++
MLIRContext::MLIRContext(Threading setting)
    : MLIRContext(DialectRegistry(), setting) {}

MLIRContext::MLIRContext(const DialectRegistry &registry, Threading setting)
    : impl(new MLIRContextImpl(setting == Threading::ENABLED &&
                               !isThreadingGloballyDisabled())) {
  // ....
}
```
