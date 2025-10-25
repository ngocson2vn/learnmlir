all:
	./build.sh

test:
	g++ -fPIC -I/usr/local/cuda-12.4/include -L. -L/usr/local/cuda-12.4/lib64 -o test test.cpp -lkernel -lcudart

clean:
	./clean.sh
