#include <iostream>
#include "backend/cuda_utils.h"

int main(int argc, char **argv) {
  int cc = cuda::getComputeCapability();
  std::cout << "GPU Compute Capability: " << cc << std::endl;

  return 0;
}
