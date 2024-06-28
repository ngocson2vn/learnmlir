all: test
	./test

expand: test.cpp
	clang++ -Xclang -ast-print -fsyntax-only test.cpp

test: test.cpp
	clang++ -o test test.cpp
