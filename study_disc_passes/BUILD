load("//mlir/util:util.bzl", "if_torch_disc") # type: ignore
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library") # type: ignore
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured") # type: ignore
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library") # type: ignore

load( # type: ignore
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "get_compatible_with_cloud",
    "tf_cc_binary",
    "tf_cc_shared_object",
    "tf_copts",
    "tf_native_cc_binary",
    "transitive_hdrs",
)

config_setting(
    name = "enable_cuda",
    values = {"define": "enable_cuda=true"},
)

########################################################################
# DISC IRs
########################################################################
gentbl_cc_library( # type: ignore
    name = "disc_ral_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(), # type: ignore
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "mlir/disc/IR/disc_ral_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "mlir/disc/IR/disc_ral_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/IR/disc_ral_ops.td",
    deps = if_torch_disc( # type: ignore
        ["@mlir-hlo//:hlo_ops_td_files"],
        ["@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"]
    ),
)

gentbl_cc_library( # type: ignore
    name = "disc_shape_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(), # type: ignore
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "mlir/disc/IR/disc_shape_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "mlir/disc/IR/disc_shape_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/IR/disc_shape_ops.td",
    deps = if_torch_disc( # type: ignore
        ["@mlir-hlo//:hlo_ops_td_files"],
        ["@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"]
    ),
)

########################################################################
# Utils
########################################################################
cc_library(
    name = "custom_call_base",
    srcs = [
        "mlir/disc/IR/custom_call_base.cc",
    ],
    hdrs = [
        "mlir/disc/IR/custom_call_base.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "mhlo_disc_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "mlir/disc/IR/hlo_disc_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "mlir/disc/IR/hlo_disc_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/IR/hlo_disc_enums.td",
    deps = if_torch_disc(
        ["@mlir-hlo//:hlo_ops_td_files"],
        ["@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"]
    )
)

gentbl_cc_library(
    name = "mhlo_disc_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "mlir/disc/IR/hlo_disc_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "mlir/disc/IR/hlo_disc_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/IR/hlo_disc_ops.td",
    deps = [
        ":mhlo_disc_enums_inc_gen",
    ] + if_torch_disc(
        ["@mlir-hlo//:hlo_ops_td_files"],
        ["@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"]
    )
)

cc_library(
    name = "mlir_hlo_delegate",
    visibility = [
        "//visibility:private"
    ],
    deps = if_torch_disc([
         "@mlir-hlo//:mlir_hlo",
         "@mlir-hlo//:legalize_to_linalg_utils",
         "@mlir-hlo//:mhlo_passes",
         "@mlir-hlo//:codegen_utils",
         "@mlir-hlo//:placement_utils",
        ], [
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:legalize_to_linalg_utils",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:codegen_utils",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:placement_utils",
        ]
    )
)

cc_library(
    name = "mlir_lhlo_delegate",
    visibility = [
        "//visibility:private"
    ],
    deps = if_torch_disc([
         "@mlir-hlo//:lhlo",
         "@mlir-hlo//:lhlo_gpu",
         "@mlir-hlo//:map_hlo_to_lhlo_op",
         "@mlir-hlo//:lmhlo_passes",
         "@mlir-hlo//:map_lmhlo_to_scalar_op",
        ], [
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lmhlo_passes",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:map_hlo_to_lhlo_op",
         "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
       ]
    )
)

cc_library(
    name = "mhlo_disc",
    srcs = [
        "mlir/disc/IR/hlo_disc_ops.cc.inc",
        "mlir/disc/IR/hlo_disc_ops.h.inc",
        "mlir/disc/IR/hlo_disc_ops.cc",
    ],
    hdrs = [
        "mlir/disc/IR/hlo_disc_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":custom_call_base",
        ":mhlo_disc_enums_inc_gen",
        ":mhlo_disc_ops_inc_gen",
        ":mlir_hlo_delegate",
        ":mlir_lhlo_delegate",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "lmhlo_disc_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "mlir/disc/IR/lhlo_disc_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "mlir/disc/IR/lhlo_disc_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/IR/lhlo_disc_enums.td",
    deps = if_torch_disc(
        ["@mlir-hlo//:hlo_ops_td_files"],
        ["@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"]
    )
)

gentbl_cc_library(
    name = "lmhlo_disc_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "mlir/disc/IR/lhlo_disc_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "mlir/disc/IR/lhlo_disc_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/IR/lhlo_disc_ops.td",
    deps = [
        ":lmhlo_disc_enums_inc_gen",
    ] + if_torch_disc(
        ["@mlir-hlo//:hlo_ops_td_files"],
        ["@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files"]
    )
)

cc_library(
    name = "lmhlo_disc",
    srcs = [
        "mlir/disc/IR/lhlo_disc_ops.cc.inc",
        "mlir/disc/IR/lhlo_disc_ops.h.inc",
        "mlir/disc/IR/lhlo_disc_ops.cc",
    ],
    hdrs = [
        "mlir/disc/IR/lhlo_disc_ops.h",
    ],
    deps = [
        ":lmhlo_disc_enums_inc_gen",
        ":lmhlo_disc_ops_inc_gen",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo_ops_structs_inc_gen",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo_structured_interface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_map_hlo_to_lhlo_op",
    hdrs = ["mlir/disc/transforms/disc_map_hlo_to_lhlo_op.h"],
    deps = [
        ":mhlo_disc",
        ":lmhlo_disc",
    ],
)

cc_library(
    name = "placement_utils",
    srcs = ["mlir/disc/transforms/placement_utils.cc"],
    hdrs = ["mlir/disc/transforms/placement_utils.h"],
    includes = ["include"],
    deps = [
        ":disc_map_hlo_to_lhlo_op",
        ":lmhlo_disc",
        ":mhlo_disc",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:map_hlo_to_lhlo_op",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cycle_detector",
    srcs = ["mlir/disc/utils/cycle_detector.cc"],
    hdrs = ["mlir/disc/utils/cycle_detector.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "disc_util",
    srcs = [
        "mlir/disc/disc_util.cc",
    ],
    hdrs = [
        "mlir/disc/disc_util.h",
    ],
    deps = [
        ":cycle_detector",
        ":placement_utils",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@org_tensorflow//tensorflow/core:lib",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
    copts=["-Wno-error=unused-result"],
)

cc_library(
    name = "pass_details",
    hdrs = [
        "mlir/disc/transforms/PassDetail.h",
    ],
    visibility = [
        "//visibility:private",  # This target is a private detail of pass implementations
    ],
    deps = [
        ":disc_shape",
        ":DiscPassIncGen",
        ":MhloDiscPassIncGen",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUPassIncGen",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "disc_shape_optimization_utils",
    srcs = [
        "mlir/disc/transforms/disc_shape_optimization_utils.cc"
    ],
    hdrs = [
        "mlir/disc/transforms/disc_shape_optimization_utils.h",
    ],
    deps = [
        ":disc_shape",
        ":disc_util",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    copts=["-Wno-error=unused-result"],
    alwayslink = 1,
)

cc_library(
    name = "shape_utils",
    srcs = ["mlir/disc/transforms/shape_utils.cc"],
    hdrs = ["mlir/disc/transforms/shape_utils.h"],
    deps = [
        ":disc_shape_optimization_utils",
        ":disc_util",
        ":mhlo_disc",
        ":pass_details",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    copts=["-Wno-error=unused-result"],
)

cc_library(
    name = "disc_supported_list",
    hdrs = ["mlir/disc/transforms/disc_supported_list.h.inc"],
)

cc_library(
    name = "source_emitter",
    srcs = ["mlir/disc/utils/source_emitter.cc"],
    hdrs = ["mlir/disc/utils/source_emitter.h"],
    includes = ["include"],
    deps = [
        ":disc_util",
        ":lmhlo_disc",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "codegen_utils",
    srcs = [
        "mlir/disc/transforms/codegen_utils.cc",
    ],
    hdrs = [
        "mlir/disc/transforms/codegen_utils.h",
    ],
    includes = [
        "tensorflow/compiler/xla/mlir_hlo/include",
    ],
    deps = [
        ":disc_shape",
        ":disc_util",
        ":source_emitter",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
    ],
    copts=["-Wno-error=unused-result"],
    alwayslink = 1,
)

cc_library(
    name = "disc_lhlo_elemental_utils",
    srcs = ["mlir/disc/transforms/lhlo_elemental_utils.cc"],
    hdrs = ["mlir/disc/transforms/lhlo_elemental_utils.h"],
    deps = [
        ":codegen_utils",
        ":disc_shape_optimization_utils",
        ":disc_util",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "fusion_utils",
    srcs = [
        "mlir/disc/transforms/fusion_utils.cc",
        "mlir/disc/transforms/fusion_utils_stitch_gpu.cc",
        "mlir/disc/transforms/fusion_utils_dot_gpu.cc",
        "mlir/disc/transforms/fusion_utils_transform_based.cc",
        "mlir/disc/transforms/fusion_utils_sparse_op_cpu.cc",
    ],
    hdrs = ["mlir/disc/transforms/fusion_utils.h"],
    deps = [
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":disc_shape_optimization_utils",
        ":disc_supported_list",
        ":disc_util",
        ":lmhlo_disc",
        ":placement_utils",
        ":shape_utils",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:placement_utils",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
    ],
)

########################################################################
# DISC passes
########################################################################
gentbl_cc_library( # type: ignore
    name = "DiscPassIncGen",
    compatible_with = get_compatible_with_cloud(), # type: ignore
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=DISC",
            ],
            "mlir/disc/transforms/disc_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/transforms/disc_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "MhloDiscPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MhloDisc",
            ],
            "mlir/disc/transforms/mhlo_disc_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlir/disc/transforms/mhlo_disc_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library( # type: ignore
    name = "disc_shape",
    srcs = [
        "mlir/disc/IR/disc_shape_ops.cc.inc",
        "mlir/disc/IR/disc_shape_ops.h.inc",
        "mlir/disc/IR/disc_shape_ops.cc",
    ],
    hdrs = [
        "mlir/disc/IR/disc_shape_ops.h",
    ],
    deps = [
        ":disc_shape_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library( # type: ignore
    name = "disc_ral",
    srcs = [
        "mlir/disc/IR/disc_ral_ops.cc.inc",
        "mlir/disc/IR/disc_ral_ops.h.inc",
        "mlir/disc/IR/disc_ral_ops.cc",
    ],
    hdrs = [
        "mlir/disc/IR/disc_ral_ops.h",
    ],
    deps = [
        ":disc_ral_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "input_inline_fusion_pattern",
    srcs = ["mlir/disc/transforms/input_inline_fusion_pattern.cc"],
    hdrs = ["mlir/disc/transforms/input_inline_fusion_pattern.h"],
    includes = ["tensorflow/compiler/xla/mlir_hlo/include"],
    deps = [
        ":codegen_utils",
        ":disc_lhlo_elemental_utils",
        ":fusion_utils",
        ":placement_utils",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:map_lmhlo_to_scalar_op",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "disc_passes",
    srcs = [
      "mlir/disc/transforms/disc_passes.cc"
    ],
    hdrs = [
        "mlir/disc/transforms/disc_passes.h",
    ],
    visibility = [
        "//visibility:private",  # This target is a private detail of pass implementations
    ],
    deps = [
        ":disc_shape",
        ":DiscPassIncGen",
        ":placement_utils",
        ":fusion_utils",
        ":disc_lhlo_elemental_utils",
        ":input_inline_fusion_pattern",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUPassIncGen",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lmhlo_passes",
    ],
)

tf_cc_binary( # type: ignore
    name = "main",
    srcs = ["main.cpp"],
    copts = tf_copts(), # type: ignore
    deps = [
        ":disc_shape",
        ":disc_ral",
        ":disc_passes",
        "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/transforms:legalize_utils",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "@org_tensorflow//tensorflow/compiler/xla/stream_executor",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:AArch64AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86Disassembler",  # fixdeps: keep
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@torch-mlir//:TorchMLIRInitAll",
        "@torch-mlir//:TorchMLIRTorchDialect",
        "@mlir-hlo//:mlir_hlo",
        "@mlir-hlo//:mhlo_passes",
        "@mlir-hlo//:lmhlo_passes",
        "@mlir-hlo//:lhlo_gpu",
    ] + if_cuda_is_configured([ # type: ignore
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_driver",
    ]),
)
