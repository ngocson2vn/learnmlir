## Argument-Dependent Lookup (ADL)
The way C++ compilers resolve function names—especially inside templates—is notoriously complex, but understanding it is essential for working with large frameworks like LLVM and MLIR.

To explain the claim, we need to look at a C++ feature called **Argument-Dependent Lookup (ADL)**, also known as Koenig Lookup, and how library authors use it to create "customization points."

### 1. Qualified vs. Unqualified Function Calls

When you call a function in C++, you can do it in two ways:

* **Qualified Call:** You explicitly tell the compiler exactly where to look by providing a namespace prefix (e.g., `llvm::hash_value(val)` or `std::swap(a, b)`). The compiler looks *only* in that specific namespace.
* **Unqualified Call:** You do not provide a prefix (e.g., `hash_value(val)` or `swap(a, b)`). The compiler now has to play detective to figure out which `hash_value` you mean.

### 2. How Unqualified Lookup Works (Enter ADL)

When the compiler sees an unqualified call like `hash_value(value)`, it performs two searches:

1.  **Ordinary Name Lookup:** It looks in the local scope, then the enclosing scopes, up to the global scope where the function is being called.
2.  **Argument-Dependent Lookup (ADL):** It looks at the **types of the arguments** being passed to the function. It then searches for the function *inside the namespaces where those argument types are defined*.

**Example of ADL in action:**
```cpp
namespace MyMath {
    struct Matrix { /* ... */ };
    void print(const Matrix& m) { /* ... */ }
}

int main() {
    MyMath::Matrix mat;
    print(mat); // UNQUALIFIED CALL. 
                // The compiler sees the argument is MyMath::Matrix.
                // It automatically searches namespace 'MyMath' and finds 'print'.
    return 0;
}
```

### 3. Why LLVM Relies on ADL for Hashing

LLVM uses a design pattern called a **Customization Point Object**. In `llvm/ADT/Hashing.h`, the authors wrote the template code to call `hash_value` *without* a prefix:

```cpp
// Inside llvm::hashing::detail...
template <typename T>
size_t get_hashable_data(const T &value) {
    // ...
    return static_cast<size_t>(hash_value(value)); // UNQUALIFIED!
}
```

**Why didn't they write `llvm::hash_value(value)`?**
Because if they forced the `llvm::` prefix, it would severely restrict extensibility. By using an unqualified call, LLVM is essentially saying: *"I don't know what type `T` is yet. But whatever it is, I will look in the namespace where `T` was defined to see if the author provided a `hash_value` function for it."*

This allows you to define your own types in your own namespaces and provide a hash function right next to them, without ever having to "inject" code into the `llvm` namespace.

### 4. Connecting This Back to Your `std::vector` Error

Let's look at exactly what happened in your code based on these rules:

1.  The MLIR tablegen output asks LLVM to hash a `std::vector<int>`.
2.  LLVM's template invokes `hash_value(value)` (unqualified).
3.  The compiler triggers **ADL**.
4.  It looks at the argument type: `std::vector<int>`.
5.  It notes that `std::vector` is defined in `namespace std`.
6.  It searches `namespace std` for a function named `hash_value`.
7.  **It finds nothing.** (Because `std::vector` does not have a `hash_value` in namespace `std`).

But wait, you wrote a `hash_value` function! Why didn't it find it? 
Because you put it in `namespace llvm`. When the compiler did ADL for `std::vector`, it explicitly searched `std`, not `llvm`. Because the LLVM header was compiled before your specific instantiation was fully visible to ordinary lookup, the compiler threw its hands up and reported an error.

### Summary

When framework authors make an unqualified function call inside a template (`hash_value(x)`), they are deliberately triggering ADL. They expect the user to define that function in the exact same namespace as the type being passed in. Because you cannot legally or safely add new functions to `namespace std`, ADL fails for standard library containers unless the hashing framework has explicitly hardcoded support for them.


## Ordinary Name Lookup
This dives into one of the most notoriously complex, yet fascinating, mechanics of C++: **Two-Phase Name Lookup** for templates. 

When I said the compiler "threw its hands up," I was referring to the strict chronological rules C++ uses to evaluate templates. To understand why your `hash_value` function was ignored, we have to look at how the compiler reads your code in two distinct phases.



### The Timeline of a Template

When the C++ compiler encounters a template, it doesn't just read it once. It processes it in two separate phases: **Phase 1 (Definition)** and **Phase 2 (Instantiation)**.

#### Phase 1: The Point of Definition
Imagine the compiler is parsing the `#include "llvm/ADT/Hashing.h"` file. It sees the implementation of the hashing templates:

```cpp
// Inside llvm/ADT/Hashing.h
template <typename T>
size_t get_hashable_data(const T &value) {
    // The compiler sees this UNQUALIFIED call.
    // It knows 'value' is dependent on the template parameter 'T'.
    return static_cast<size_t>(hash_value(value)); 
}
```

During this first phase, the compiler performs **Ordinary Name Lookup** for the dependent name `hash_value`. It essentially takes a "snapshot" of every `hash_value` function that is visible *at this exact line of code in the LLVM header*. 

Crucially, **your `Dialect.h` has not been included or processed yet** from the perspective of this LLVM header. Therefore, your custom `llvm::hash_value(std::vector<int>)` is not in the snapshot.

#### Phase 2: The Point of Instantiation
Later on in your compilation unit (likely when compiling your MLIR dialect's `.cpp` file), the compiler reaches `ExampleAttributes.cpp.inc`. It sees the call to `llvm::hash_combine` with a `std::vector<int>`, which forces it to actually generate the code for the template.

```cpp
// Inside ExampleAttributes.cpp.inc
return ::llvm::hash_combine(std::get<0>(tblgenKey), ...); 
// The compiler says: "Okay, I need to instantiate the template for T = std::vector<int>"
```

Now, the compiler enters Phase 2. It needs to resolve that `hash_value(value)` call it saved from Phase 1. Here is the golden rule of C++ Two-Phase Lookup:

At the Point of Instantiation (Phase 2), the compiler builds a candidate list of functions using **only two sources**:
1.  **From Ordinary Lookup:** The functions it took a snapshot of during **Phase 1** (at the Point of Definition).
2.  **From ADL:** Functions found by searching the namespaces of the arguments (in this case, `namespace std` for `std::vector`) at the point of instantiation.

### Why Your Code Failed

Let's look at where your custom function falls in this rulebook:

* **Did it make the Phase 1 snapshot?** No. `llvm/ADT/Hashing.h` knows nothing about your `Dialect.h`. So, Ordinary Name Lookup fails.
* **Did it get found by Phase 2 ADL?** No. The argument is `std::vector<int>`, so ADL searches `namespace std`. You put your function in `namespace llvm`. So, ADL fails.

Because it failed both allowed lookup methods, the compiler throws the exact error you saw: *"call to function 'hash_value' that is neither visible in the template definition [Phase 1] nor found by argument-dependent lookup [Phase 2]"*.

### The "Ah-Ha!" Moment

Many C++ developers intuitively think: *"I included `Dialect.h` before the compiler instantiated the template in the `.cpp` file, so the compiler should just know about my function!"*

But C++ is explicitly designed *not* to work that way. If Ordinary Name Lookup at the point of instantiation (Phase 2) were allowed to resolve template calls, the behavior of a template would change wildly depending on what random headers you happened to include before calling it. It would be a nightmare to debug. By restricting Ordinary Lookup to the Point of Definition (Phase 1), C++ guarantees that templates behave predictably.

### How the Firewall Blocks Your Code
C++ enforces a strict firewall: **A template will only ever see normal functions that were visible to the template's author at the exact moment they wrote the template.**

Let's map this firewall onto your specific files:

**1. `llvm/ADT/Hashing.h` (The Point of Definition)**
* The compiler reads the LLVM template `get_hashable_data`. 
* It sees `hash_value(value)`.
* **The Firewall is Set:** The compiler does Ordinary Name Lookup *right here*. It locks in a snapshot of all `hash_value` functions that exist at this exact microsecond. Your function does not exist yet.

**2. `Dialect.h` (Your Definition)**
* You define `llvm::hash_value(const std::vector<T>&)`.
* For any *normal* code written below this line, Ordinary Name Lookup will find your function perfectly.

**3. `ExampleAttributes.cpp.inc` (The Point of Instantiation)**
* The compiler actually builds the template code for `std::vector<int>`.
* It needs to resolve `hash_value(value)`.
* **The Rules apply:** C++ says, *"Look at the Phase 1 snapshot. Is it there? No. Okay, now perform Argument-Dependent Lookup (ADL) on `std::vector` (which checks `namespace std`). Is it there? No."*
* Even though your `hash_value` definition is physically sitting right there in the compiler's memory, **Ordinary Name Lookup is disabled at this step**. The compiler acts as if your function doesn't exist for the purpose of this template.
