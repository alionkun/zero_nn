#include "lab/2/tensor.h"

namespace zeronn {
namespace lab2 {

Tensor& Tensor::operator+(const Tensor& rhs) const {
  auto result = MakeIntermediateTensor(data_ + rhs.data_);
  result->parent_nodes_.push_back(const_cast<Tensor*>(this));
  result->parent_nodes_.push_back(const_cast<Tensor*>(&rhs));
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return 1.0 * g; }
      );
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return 1.0 * g; }
      );
  return *result;
}

Tensor& Tensor::operator-(const Tensor& rhs) const {
  auto result = MakeIntermediateTensor(data_ - rhs.data_);
  result->parent_nodes_.push_back(const_cast<Tensor*>(this));
  result->parent_nodes_.push_back(const_cast<Tensor*>(&rhs));
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return 1.0 * g; }
      );
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return -1.0 * g; }
      );
  return *result;
}

Tensor& Tensor::operator*(const Tensor& rhs) const {
  auto result = MakeIntermediateTensor(data_ * rhs.data_);
  result->parent_nodes_.push_back(const_cast<Tensor*>(this));
  result->parent_nodes_.push_back(const_cast<Tensor*>(&rhs));
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return rhs.data_ * g; }
      );
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return data_ * g; }
      );
  return *result;
}

Tensor& Tensor::operator/(const Tensor& rhs) const {
  auto result = MakeIntermediateTensor(data_ / rhs.data_);
  result->parent_nodes_.push_back(const_cast<Tensor*>(this));
  result->parent_nodes_.push_back(const_cast<Tensor*>(&rhs));
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return 1.0 / rhs.data_ * g; }
      );
  result->parent_node_gradient_functions_.push_back(
        [&](float g) { return data_ * g; }
      );
  return *result;
}


std::vector<Tensor*> Tensor::intermediate_tensors_;


} // namespace lab2
} // namespace zeronn

