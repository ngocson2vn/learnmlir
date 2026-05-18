// ADL kicks in 
namespace llvm {

template <typename Container>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Container& arr) {
  if (arr.empty()) {
    os << "[]";
    return os;
  }

  os << "[" << arr[0];
  for (int i = 1; i < arr.size(); i++) {
    os << ", " << arr[i];
  }
  os << "]";

  return os;
}

}
