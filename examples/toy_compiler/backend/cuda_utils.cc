#include <string>
#include <iostream>
#include <filesystem>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "cuda_utils.h"

namespace fs = std::filesystem;

namespace cuda {

status::Result<int> getComputeCapability() {
  int dev = 0;
  cudaError_t err = cudaSetDevice(dev);
  if (err != cudaError::cudaSuccess) {
    return {
      0,
      cudaGetErrorString(err)
    };
  }

  cudaDeviceProp deviceProp;
  err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaError::cudaSuccess) {
    return {
      0,
      cudaGetErrorString(err)
    };
  }

  return deviceProp.major + deviceProp.minor;
}

status::Result<int> ParseCudaArch(const std::string& arch_str) {
  auto prefixPos = arch_str.find("sm_");
  if (prefixPos == std::string::npos) {
    return {
      0,
      "Could not parse cuda architecture prefix (expected sm_)"
    };
  }

  return std::stoi(arch_str.substr(3, arch_str.size() - 3));
}

status::Result<std::string> getCudaDir() {
  fs::path cudaDir = toy::utils::getStrEnv("CUDA_DIR");
  if (cudaDir.empty()) {
    return {
      "",
      "${CUDA_DIR} is unset!"
    };
  }

  if (!fs::exists(cudaDir)) {
    return {
      "",
      std::string("${CUDA_DIR} ").append(cudaDir.string())
                                 .append(" does not exist!")
    };
  }

  return cudaDir.string();
}

std::string getPtxasPath() {
  auto cudaDir = getCudaDir();
  return cudaDir.value() + "/bin/ptxas";
}

std::string getSupportedPtxVersion() {
  int cudaRuntimeVersion;
  cudaError_t err = cudaRuntimeGetVersion(&cudaRuntimeVersion);
  if (err != cudaError::cudaSuccess) {
    return std::string("");
  }

  int major = cudaRuntimeVersion / 1000;
  int minor = (cudaRuntimeVersion % 1000) / 10;

  if (major >= 11) {
    major -= 4;
  }

  std::string ptxVersion = std::to_string(major).append(std::to_string(minor));
  std::cout << "ptxVersion: " << ptxVersion << std::endl;

  return ptxVersion;
}

std::string getLibdevice() {
  auto cudaDir = std::getenv("CUDA_DIR");
  return std::string(cudaDir).append("/nvvm/libdevice/libdevice.10.bc");
}

} // namespace cuda
