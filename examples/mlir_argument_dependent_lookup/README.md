# MLIR and Argument-dependent
The C++ compiler reports the following error:
```C++
FAILED: [code=1] CMakeFiles/ExampleIR.dir/Dialect.cpp.o 
CCACHE_CPP2=yes CCACHE_HASHDIR=yes CCACHE_SLOPPINESS=pch_defines,time_macros /usr/bin/ccache /usr/bin/clang++  -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/mlir/include -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/build/include -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/build/tools/mlir/include -I/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup -I/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/build -fno-rtti -g -std=gnu++17 -MD -MT CMakeFiles/ExampleIR.dir/Dialect.cpp.o -MF CMakeFiles/ExampleIR.dir/Dialect.cpp.o.d -o CMakeFiles/ExampleIR.dir/Dialect.cpp.o -c /data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/Dialect.cpp
In file included from /data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/Dialect.cpp:3:
In file included from /data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/Dialect.h:4:
/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include/llvm/ADT/Hashing.h:361:32: error: call to function 'hash_value' that is neither visible in the template definition nor found by argument-dependent lookup
  361 |     return static_cast<size_t>(hash_value(value));
      |                                ^
/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include/llvm/ADT/Hashing.h:547:63: note: in instantiation of function template specialization 'llvm::hashing::detail::get_hashable_data<std::vector<int>>' requested here
  547 |     buffer_ptr = combine_data(length, buffer_ptr, buffer_end, get_hashable_data(arg));
      |                                                               ^
/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include/llvm/ADT/Hashing.h:595:17: note: in instantiation of function template specialization 'llvm::hashing::detail::hash_combine_recursive_helper::combine<std::vector<int>, int>' requested here
  595 |   return helper.combine(0, helper.buffer, helper.buffer + 64, args...);
      |                 ^
/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/build/ExampleAttributes.cpp.inc:104:20: note: in instantiation of function template specialization 'llvm::hash_combine<std::vector<int>, int>' requested here
  104 |     return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
      |                    ^
/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/Dialect.h:9:33: note: 'hash_value' should be declared prior to the call site
    9 | template <typename T> hash_code hash_value(const std::vector<T>& vec) {
      |                                 ^
1 error generated.
[21/23] CCACHE_CPP2=yes CCACHE_HASHDIR=yes CCACHE_SLOPPINESS=pch_defines,time_macros /usr/bin/ccache /usr/bin/clang++  -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/mlir/include -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/build/include -I/data00/home/son.nguyen/workspace/learnmlir/llvm-project/build/tools/mlir/include -I/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup -I/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/build -fno-rtti -g -std=gnu++17 -g -O0 -MD -MT CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /data00/home/son.nguyen/workspace/learnmlir/examples/mlir_argument_dependent_lookup/main.cpp
ninja: build stopped: subcommand failed.

make: *** [Makefile:2: all] Error 1
```

The error is related to [ADL](../../docs/mlir_argument_dependent_lookup.md).

