#include <cuda_bf16.h>
#include <iostream>

__global__ void embeddingGatherKernel(int *gpu_input_tokens,
                                      __nv_bfloat16 *input_embeddings,
                                      __nv_bfloat16 *embed_tokens) {
  int token_index = blockIdx.x / 2;
  int component = (blockIdx.x % 2) * 1024 + threadIdx.x;
  int token_id = gpu_input_tokens[token_index];

  input_embeddings[token_index * 2048 + component] = embed_tokens[token_id * 2048 + component];
};

void embeddingGather(
    int *gpu_input_tokens, // {32,34,12,...}: list of input token ids
    __nv_bfloat16 *out_input_embeddings, // output: N 2048-length vectors.
                                         // Embeddings for the input tokens
    __nv_bfloat16 *embed_tokens,         // all 128k embeddings
    int num_input_tokens) {
  embeddingGatherKernel<<<num_input_tokens * 2, 1024>>>(
      gpu_input_tokens, out_input_embeddings, embed_tokens);
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::cerr << "CUDA launch failed: " << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
