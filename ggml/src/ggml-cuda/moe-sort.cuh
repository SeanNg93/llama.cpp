#pragma once

#include "common.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>

// Kernel to build sort keys and values from ids tensor
static __global__ void moe_build_sort_keys(
    const int32_t * __restrict__ ids,
    int64_t ids_nb0,
    int64_t ids_nb1,
    int32_t * __restrict__ keys,
    int32_t * __restrict__ values,
    int32_t * __restrict__ orig_idx,
    int64_t ne12,
    int64_t n_expert_used,
    int64_t ne11)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ne12 * n_expert_used) {
        return;
    }

    const int64_t i12 = idx / n_expert_used;
    const int64_t iex = idx % n_expert_used;

    const int32_t expert = *(const int32_t *)((const char *)ids + i12 * ids_nb1 + iex * ids_nb0);

    keys[idx]     = expert;
    values[idx]   = (int32_t)(i12 * ne11 + iex % ne11);
    orig_idx[idx] = (int32_t)idx;
}

// Kernel to build inverse permutation
static __global__ void moe_build_inverse_perm(
    const int32_t * __restrict__ sorted_orig_idx,
    int32_t * __restrict__ ids_from_sorted,
    int64_t n)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ids_from_sorted[sorted_orig_idx[i]] = (int32_t)i;
}

// Kernel to count tokens per expert using binary search on sorted keys
static __global__ void moe_count_tokens_per_expert(
    const int32_t * __restrict__ sorted_keys,
    int32_t * __restrict__ tokens_per_expert,
    int64_t n_total,
    int64_t n_experts)
{
    const int64_t expert = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= n_experts) {
        return;
    }

    // Binary search for first occurrence of this expert
    int64_t left = 0, right = n_total;
    while (left < right) {
        int64_t mid = (left + right) / 2;
        if (sorted_keys[mid] < expert) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    const int64_t first = left;

    // Binary search for first element > expert
    right = n_total;
    while (left < right) {
        int64_t mid = (left + right) / 2;
        if (sorted_keys[mid] <= expert) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    tokens_per_expert[expert] = (int32_t)(left - first);
}

// Main function: GPU-based MoE sorting using CUB radix sort
inline void moe_sort_cuda(
    ggml_cuda_pool & pool,
    const int32_t * ids,
    int64_t ids_nb0,
    int64_t ids_nb1,
    int32_t * ids_to_sorted,
    int32_t * ids_from_sorted,
    int32_t * tokens_per_expert_dev,
    int64_t ne12,
    int64_t n_expert_used,
    int64_t ne11,
    int64_t n_experts,
    cudaStream_t stream)
{
    const int64_t n_total = ne12 * n_expert_used;
    constexpr int block_size = 256;
    const int grid_size = (n_total + block_size - 1) / block_size;

    // Allocate temporary buffers for sorting
    ggml_cuda_pool_alloc<int32_t> keys_in(pool, n_total);
    ggml_cuda_pool_alloc<int32_t> values_in(pool, n_total);
    ggml_cuda_pool_alloc<int32_t> orig_idx_in(pool, n_total);
    ggml_cuda_pool_alloc<int32_t> keys_out(pool, n_total);
    ggml_cuda_pool_alloc<int32_t> orig_idx_out(pool, n_total);

    // Build keys, values, and original indices
    moe_build_sort_keys<<<grid_size, block_size, 0, stream>>>(
        ids, ids_nb0, ids_nb1,
        keys_in.ptr, values_in.ptr, orig_idx_in.ptr,
        ne12, n_expert_used, ne11);

    // Sort keys and values to get ids_to_sorted
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        keys_in.ptr, keys_out.ptr,
        values_in.ptr, ids_to_sorted,
        n_total, 0, 32, stream);

    ggml_cuda_pool_alloc<uint8_t> temp_storage(pool, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_storage.ptr, temp_storage_bytes,
        keys_in.ptr, keys_out.ptr,
        values_in.ptr, ids_to_sorted,
        n_total, 0, 32, stream);

    // Sort keys and orig_idx to build inverse permutation
    size_t temp_storage_bytes2 = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes2,
        keys_in.ptr, keys_out.ptr,
        orig_idx_in.ptr, orig_idx_out.ptr,
        n_total, 0, 32, stream);

    if (temp_storage_bytes2 > temp_storage_bytes) {
        temp_storage.~ggml_cuda_pool_alloc();
        new (&temp_storage) ggml_cuda_pool_alloc<uint8_t>(pool, temp_storage_bytes2);
    }

    cub::DeviceRadixSort::SortPairs(
        temp_storage.ptr, temp_storage_bytes2,
        keys_in.ptr, keys_out.ptr,
        orig_idx_in.ptr, orig_idx_out.ptr,
        n_total, 0, 32, stream);

    // Build inverse permutation
    const int inv_grid = (n_total + block_size - 1) / block_size;
    moe_build_inverse_perm<<<inv_grid, block_size, 0, stream>>>(
        orig_idx_out.ptr, ids_from_sorted, n_total);

    // Count tokens per expert
    const int expert_grid = (n_experts + block_size - 1) / block_size;
    moe_count_tokens_per_expert<<<expert_grid, block_size, 0, stream>>>(
        keys_out.ptr, tokens_per_expert_dev, n_total, n_experts);
}

#endif // GGML_CUDA_USE_CUB
