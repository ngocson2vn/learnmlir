"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "fdac4c4e92e5a83ac5e4fa6d1d2970c0c4df8fa8"
    LLVM_SHA256 = "df274c0e7c218f833e69b6ae5abce3000540f84f557fa3f4e57608ddae336351"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = [
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )

def tf_repo(name):
    """Imports TF."""
    TF_COMMIT = "12b64405765dd183ef72ba3e3fd06e7bad14e034"
    TF_SHA256 = "0b2c1a71eb2aa81cc696f016d48c21c7ed31917c198ce92f1c580f9c6a3f5c3b"

    tf_http_archive(
        name = name,
        sha256 = TF_SHA256,
        strip_prefix = "tensorflow-{commit}".format(commit = TF_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/tensorflow/tensorflow/archive/{commit}.tar.gz".format(commit = TF_COMMIT),
            "https://github.com/tensorflow/tensorflow/archive/{commit}.tar.gz".format(commit = TF_COMMIT),
        ],
        build_file = "//third_party/tensorflow:tensorflow.BUILD",
        patch_file = [],
        link_files = {},
    )
