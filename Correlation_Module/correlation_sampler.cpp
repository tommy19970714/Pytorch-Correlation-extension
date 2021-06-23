#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <torch/script.h>

// declarations

torch::Tensor correlation_cpp_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int64_t kH, int64_t kW,
    int64_t patchH, int64_t patchW,
    int64_t padH, int64_t padW,
    int64_t dilationH, int64_t dilationW,
    int64_t dilation_patchH, int64_t dilation_patchW,
    int64_t dH, int64_t dW);

std::vector<torch::Tensor> correlation_cpp_backward(
    torch::Tensor grad_output,
    torch::Tensor input1,
    torch::Tensor input2,
    int64_t kH, int64_t kW,
    int64_t patchH, int64_t patchW,
    int64_t padH, int64_t padW,
    int64_t dilationH, int64_t dilationW,
    int64_t dilation_patchH, int64_t dilation_patchW,
    int64_t dH, int64_t dW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_cpp_forward, "Spatial Correlation Sampler Forward");
  m.def("backward", &correlation_cpp_backward, "Spatial Correlation Sampler backward");
}

TORCH_LIBRARY(my_ops, m) {
  m.def("forward", &correlation_cpp_forward);
  m.def("backward", &correlation_cpp_backward);
}
