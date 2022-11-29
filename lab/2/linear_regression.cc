#include "lab/2/tensor.h"
#include <random>
#include <glog/logging.h>

using zeronn::lab2::Tensor;

const float kRealA = 3.14;
const float kRealB = -17.99;

void GenerateDataset(std::vector<float>& Xs, std::vector<float>& Ys, int N) {
  Xs.resize(N);
  Ys.resize(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int i = 0; i < N; ++i) {
    float x = dis(gen);
    float y = kRealA * x + kRealB;
    y += dis(gen) * 0.005;
    Xs[i] = x;
    Ys[i] = y;
  }
}

int main() {
  // generate dataset
  const int N = 100;
  std::vector<float> Xs, Ys;
  GenerateDataset(Xs, Ys, N);
  // learning
  const int epochs = 100;
  const float lr = 0.001;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  Tensor A(dis(gen));
  Tensor B(dis(gen));
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0;
    for (int i = 0; i < N; ++i) {
      // forward pass
      float x = Xs[i];
      float y_true = Ys[i];
      Tensor X(x);
      Tensor Y_TRUE(y_true);
      auto Y_PRED = A * X + B;
      auto LOSS = (Y_TRUE - Y_PRED) * (Y_TRUE - Y_PRED);
      epoch_loss += LOSS.data_;
      // backward pass
      LOSS.ResetGradient();
      LOSS.Backward();
      // SGD
      A.data_ -= lr * A.gradient_;
      B.data_ -= lr * B.gradient_;
      // clean up
      Tensor::ReleaseIntermediateTensors();
    }
    LOG(INFO) << "epoch=" << epoch << "/" << epochs << ", loss=" << epoch_loss / N << ", A=" << A.data_ << ", B=" << B.data_;
  }

  LOG(INFO) << "kRealA=" << kRealA << ", kRealB=" << kRealB;
  LOG(INFO) << "learned A=" << A.data_ << ", learned B=" << B.data_;

  return 0;
}

