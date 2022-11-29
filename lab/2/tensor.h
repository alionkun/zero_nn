#ifndef _ZERONN_LAB2_TENSOR_
#define _ZERONN_LAB2_TENSOR_

#include <vector>
#include <functional>


namespace zeronn {
namespace lab2 {

class Tensor {
public:
  explicit Tensor(float data) {
    data_ = data;
  }

  Tensor() = delete;
  Tensor(const Tensor& other) = default; // or delete?
  Tensor(Tensor&& other) = delete;

  Tensor& operator+(const Tensor& rhs) const; // return a reference bcs we own all intermediate tensors
  Tensor& operator-(const Tensor& rhs) const;
  Tensor& operator*(const Tensor& rhs) const;
  Tensor& operator/(const Tensor& rhs) const;

  void ResetGradient(bool back=true) {
    gradient_ = 0.0;
    if (back) {
      for (auto p : parent_nodes_) {
        p->ResetGradient();
      }
    }
  }
  
  void Backward() {
    Backward(1.0);
  }

  void Backward(float g) {
    gradient_ += g;
    for (size_t i = 0; i < parent_nodes_.size(); ++i) {
      parent_nodes_[i]->Backward(parent_node_gradient_functions_[i](g));
    }
  }

  static size_t ReleaseIntermediateTensors() {
    size_t count = intermediate_tensors_.size();
    for (auto tensor : intermediate_tensors_) {
      delete tensor;
    }
    intermediate_tensors_.clear();
    return count;
  }

  float data_;
  float gradient_ = 0.0;
  std::vector<Tensor*> parent_nodes_;
  using GradientFunction = std::function<float(float)>;
  std::vector<GradientFunction> parent_node_gradient_functions_;

private:
  static std::vector<Tensor*> intermediate_tensors_;

  static Tensor* MakeIntermediateTensor(float data) {
    auto tensor = new Tensor(data);
    intermediate_tensors_.push_back(tensor);
    return tensor;
  }

};

} // namespace lab2
} // namespace zeronn


#endif // _ZERONN_LAB2_TENSOR_
