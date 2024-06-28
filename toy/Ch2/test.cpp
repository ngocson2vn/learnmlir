// #include <iostream>
#include <cstdio>
#include <initializer_list>

class Dialect;

class Op1 {
public:
  Op1() {

  }
};

class Op2 {
public:
  Op2() {

  }
};

class RegisteredOperationName {
public:
  /// Register a new operation in a Dialect object.
  /// This constructor is used by Dialect objects when they register the list of
  /// operations they contain.
  template <typename T>
  static void insert(Dialect &dialect) {
    // Do nothing
  }
};

class Dialect {
public:
  /// This method is used by derived classes to add their operations to the set.
  ///
  template <typename... Args>
  void addOperations() {
    // This initializer_list argument pack expansion is essentially equal to
    // using a fold expression with a comma operator. Clang however, refuses
    // to compile a fold expression with a depth of more than 256 by default.
    // There seem to be no such limitations for initializer_list.
    auto x = (RegisteredOperationName::insert<Op1>(*this), 10);
    printf("%d\n\n", x);

    (void)std::initializer_list<int>{0, (RegisteredOperationName::insert<Args>(*this), 0)...};
  }
};

class DialectRegistrar {
public:
  DialectRegistrar() {
    Dialect d;
    d.addOperations<Op1, Op2>();
  }
};

static DialectRegistrar dr;
int main(int argc, char** argv) {
  printf("Just a test program\n");
}
