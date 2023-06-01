licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "sort_functions_test",
    srcs = ["sort_functions_test.cc"],
    copts = ["-std=c++17"],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
