/*
 * @Author: Jiaqi Gu
 * @Date: 2020-06-04 14:16:01
 * @LastEditors: Jiaqi Gu && jiaqigu@asu.edu
 * @LastEditTime: 2025-03-06 01:28:20
 */
#include <bits/stdc++.h>
#include <torch/torch.h>
using namespace std;
using namespace torch::indexing;

template <typename T>
int cuDSSSpsolveCUDALauncher(int *csr_offsets_d, int *csr_columns_d,
                             T *csr_values_d_raw, T *b_values_d_raw,
                             T *x_values_d_raw, int n, int nrhs, int nnz,
                             int _mtype);

torch::Tensor cudss_spsolve_forward(torch::Tensor csr_offsets,
                                    torch::Tensor csr_columns,
                                    torch::Tensor csr_values,
                                    torch::Tensor b_values, int _mtype) {
  TORCH_CHECK(csr_offsets.is_contiguous(), "csr_offset must be contiguous")
  TORCH_CHECK(csr_columns.is_contiguous(), "csr_columns must be contiguous")
  TORCH_CHECK(csr_values.is_contiguous(), "csr_values must be contiguous")
  TORCH_CHECK(b_values.is_contiguous(), "b_values must be contiguous")

  int nrhs = 0;
  if (b_values.dim() == 1) {
    nrhs = 1;
  } else {
    nrhs = b_values.size(0);
  }
  int n = b_values.size(-1) / 2;
  int nnz = csr_values.size(0) / 2;
  auto x_values = torch::zeros_like(b_values);
  AT_DISPATCH_FLOATING_TYPES(
      b_values.scalar_type(), "cuDSSSpsolveCUDALauncher", ([&] {
        cuDSSSpsolveCUDALauncher<scalar_t>(
            csr_offsets.data_ptr<int>(), csr_columns.data_ptr<int>(),
            csr_values.data_ptr<scalar_t>(), b_values.data_ptr<scalar_t>(),
            x_values.data_ptr<scalar_t>(), n, nrhs, nnz, _mtype);
      }));
  return x_values;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cudss_spsolve", &cudss_spsolve_forward,
        "complex symmetric sparse linear solve");
}
