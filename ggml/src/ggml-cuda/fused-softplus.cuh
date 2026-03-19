#pragma once
#include "common.cuh"

void ggml_cuda_op_fused_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * add_src0, ggml_tensor * add_src1);
