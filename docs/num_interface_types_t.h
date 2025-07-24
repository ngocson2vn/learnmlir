// examples/bazel-examples/external/llvm-project/mlir/include/mlir/Support/InterfaceSupport.h
template <typename T, typename... Args>
using has_get_interface_id = decltype(T::getInterfaceID());

template <typename T>
using detect_get_interface_id = llvm::is_detected<has_get_interface_id, T>; // Refer to ./is_detected.h

/// Template utility that computes the number of elements within `T` that
/// satisfy the given predicate.
template <template <class> class Pred, size_t N, typename... Ts>
struct count_if_t_impl : public std::integral_constant<size_t, N> {};

template <template <class> class Pred, size_t N, typename T, typename... Us>
struct count_if_t_impl<Pred, N, T, Us...>
    : public std::integral_constant<
          size_t,
          count_if_t_impl<Pred, N + (Pred<T>::value ? 1 : 0), Us...>::value> {};

template <template <class> class Pred, typename... Ts>
using count_if_t = count_if_t_impl<Pred, 0, Ts...>;

template <typename... Types>
using num_interface_types_t = count_if_t<detect_get_interface_id, Types...>;
