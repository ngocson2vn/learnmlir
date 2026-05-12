llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SmallVector<int64_t>& vec) {
  if (vec.empty()) {
    os << "[]";
    return os;
  }

  os << "[" << vec[0];
  for (int i = 1; i < vec.size(); i++) {
    os << ", " << vec[i];
  }
  os << "]";

  return os;
}
