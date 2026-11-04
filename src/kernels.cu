#include <cuda_bf16.h>
#include <iostream>

__global__ void embeddingGatherKernel(int *gpu_input_tokens,
                                      __nv_bfloat16 *input_embeddings,
                                      __nv_bfloat16 *embed_tokens) {
  int workIndex = threadIdx.x + blockIdx.x * 1024;
  input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x / 2] * 2048 + threadIdx.x];
};

void embeddingGather(
    int *gpu_input_tokens, // {32,34,12,...}: list of input token ids
    __nv_bfloat16 *out_input_embeddings, // output: N 2048-length vectors.
                                         // Embeddings for the input tokens
    __nv_bfloat16 *embed_tokens,         // all 128k embeddings
    int num_input_tokens
) {
  embeddingGatherKernel<<<num_input_tokens*2, 1024>>>(gpu_input_tokens, out_input_embeddings, embed_tokens);
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
      std::cerr << "CUDA launch failed: "
                << cudaGetErrorString(error) << '\n';
      std::exit(EXIT_FAILURE);
  }
}
