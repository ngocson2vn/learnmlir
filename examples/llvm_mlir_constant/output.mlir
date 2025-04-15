; ModuleID = 'sony'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define i32 @main(i32 %0) {
  ret i32 0
}
