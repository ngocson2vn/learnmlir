# Lower MLIR builtin dialect to LLVM IR
This project demonstrates lowering a `mlir::ModuleOp` to LLVM IR<br/>

## Prerequisites
Install clang-17 and lld-17
```Bash
sudo apt install clang-17
sudo apt install lld-17

sudo ln -sf /usr/lib/llvm-17/bin/clang /usr/bin/clang
sudo ln -sf /usr/lib/llvm-17/bin/clang++ /usr/bin/clang++
sudo ln -sf /usr/lib/llvm-17/bin/ld.lld /usr/bin/ld.lld
```

## How to build
```Bash
./build.sh
```
LLVM and MLIR libs will be built automatically
<br/>

## How to run
```Bash
./run.sh
```
Output:
```
=================================================
Original moduleOp
=================================================
module {
  llvm.func @main(%arg0: i32) -> i32 {
    %0 = llvm.add %arg0, %arg0 : i32
    llvm.return %0 : i32
  }
}
=================================================

=================================================
LLVM IR
=================================================
; ModuleID = 'sony'
source_filename = "LLVMDialectModule"

define i32 @main(i32 %0) {
  %2 = add i32 %0, %0
  ret i32 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
=================================================
```
<br/>
