all: test
	./test

expand: test.cpp
	clang++ -Xclang -ast-print -fsyntax-only test.cpp

test: test.cpp
	clang++ -g -O0 -o test test.cpp
