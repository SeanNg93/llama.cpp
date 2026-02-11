#include "common.cuh"

void ggml_cuda_solve_tri(ggml_backend_cuda_context & ctx,
                         const float * A, const float * B, float * X,
                         int n, int k,
                         int64_t ne02, int64_t ne03,
                         size_t nb02, size_t nb03,
                         size_t nb12, size_t nb13,
                         size_t nb2,  size_t nb3);

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
