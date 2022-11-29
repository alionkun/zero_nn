#include "lab/2/tensor.h"
#include <gtest/gtest.h>

using zeronn::lab2::Tensor;

TEST(Tensor, Gradient) {
  // y = a * x * x + b * x + c
  // by hand
  float a = 3.1;
  float x = 2.0;
  float b = -1.2;
  float c = 99.0;
  float y = a * x * x + b * x + c;
  float dy_da = x * x;
  float dy_dx = 2.0 * a * x + b;
  float dy_db = x;
  float dy_dc = 1.0;
  // by autograd
  Tensor A(a);
  Tensor X(x);
  Tensor B(b);
  Tensor C(c);
  auto Y = A * X * X + B * X + C;
  EXPECT_EQ(Y.data_, y);
  Y.Backward();
  EXPECT_EQ(A.gradient_, dy_da);
  EXPECT_EQ(X.gradient_, dy_dx);
  EXPECT_EQ(B.gradient_, dy_db);
  EXPECT_EQ(C.gradient_, dy_dc);
  // reset gradient
  Y.ResetGradient();
  Y.Backward();
  EXPECT_EQ(A.gradient_, dy_da);
  EXPECT_EQ(X.gradient_, dy_dx);
  EXPECT_EQ(B.gradient_, dy_db);
  EXPECT_EQ(C.gradient_, dy_dc);
  // release intermediate tensors
  EXPECT_EQ(Tensor::ReleaseIntermediateTensors(), 5);

}
