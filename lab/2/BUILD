
cc_library(
  name = "tensor",
  srcs = [
    "tensor.cc"
  ],
  hdrs = [
    "tensor.h"
  ],
  visibility = [
    "//visibility:public"
  ],
  copts = [
    "-std=c++17"
  ]
)

cc_test(
  name = "gradient_test",
  srcs = [
    "gradient_test.cc",
  ],
  deps = [
    "//lab/2:tensor",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_binary(
  name = "linear_regression",
  srcs = [
    "linear_regression.cc",
  ],
  deps = [
    "//lab/2:tensor",
    "@com_github_google_glog//:glog"
  ]
)

