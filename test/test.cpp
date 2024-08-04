#include <cstdio>
#include <assert.h>
#include <utility>
#include <typeinfo>
#include <cstdint>

#define LOG(...) printf("%s\n", #__VA_ARGS__)
#define LOGF(format, ...) printf(format, __VA_ARGS__)

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

template <typename T>
class TypeDetector;

int main(int argc, char** argv) {
  // int a = 1;
  // int b = 2;
  // assert(a == b && "dialect ctor failed");

  f2(Dummy(), 1, "Hello");
  int* pInt = new int(100);
  LOGF("%d\n", *pInt);

  auto one = 1L;
  LOGF("type of (1L) = %s\n", typeid(one).name());
  LOGF("size of (1L) = %ld bytes\n", sizeof(one));

  auto mask = (sizeof(int64_t) * 8 - 1);
  LOGF("mask = %ld\n", mask);
}