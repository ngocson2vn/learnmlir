#include <cstdio>
#include <assert.h>
#include <utility>

#define LOG(...) printf("%s\n", #__VA_ARGS__)

struct Dummy {
  int data = 0;
};

template <typename... InitType>
void f1(InitType&&... initVal) {
  LOG(initVal...);
}

template <typename... Type>
void f2(Type&&... val) {
  f1(std::forward<Type>(val)...);
}

int main(int argc, char** argv) {
  // int a = 1;
  // int b = 2;
  // assert(a == b && "dialect ctor failed");

  f2(Dummy(), 1, "Hello");
}