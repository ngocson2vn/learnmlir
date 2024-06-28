#!/bin/bash

clang++ -Xclang -ast-print -fsyntax-only test.cpp
clang++ -o test test.cpp