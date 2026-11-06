#include <cuda_bf16.h>
#include <iostream>

__global__ void embeddingGatherKernel(int *gpu_input_tokens,
                                      __nv_bfloat16 *input_embeddings,
                                      __nv_bfloat16 *embed_tokens) {
  int workIndex = threadIdx.x + blockIdx.x * 2048;
  input_embeddings[workIndex] =
      embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
  input_embeddings[workIndex + 1024] =
      embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
};

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output,
                              __nv_bfloat16 *norm_weights) {
  __shared__ float rms_vector[1024];
  int workIndex = threadIdx.x + blockIdx.x * 2048;
  rms_vector[threadIdx.x] =
      (float)input[workIndex] * (float)input[workIndex] +
      (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
  __syncthreads();

  if (threadIdx.x == 0) {
    float total = 0;
    for (int i = 0; i < 1024; i++) {
      total += rms_vector[i];
    }
    // total = sum (ai**2)

    int offset = blockIdx.x * 2048;
    for (int i = offset; i < 2048 + offset; i++) {
        output[i] = input[i] / __float2bfloat16(sqrt(total / 2048.0 + 1.0e-5));
        printf("!! %f", __bfloat162float(output[i]));
    }

    printf("{%f}", total);
  }
}

void embeddingGather(
    int *gpu_input_tokens,           // {32,34,12,...}: list of input token ids
    __nv_bfloat16 *input_embeddings, // output: N 2048-length vectors.
                                     // Embeddings for the input tokens
    __nv_bfloat16 *embed_tokens,     // all 128k embeddings
    int num_input_tokens) {
  embeddingGatherKernel<<<num_input_tokens, 1024>>>(
      gpu_input_tokens, input_embeddings, embed_tokens);

  __nv_bfloat16 *hidden_state; // N 2048-length vectors (but normalized)
  cudaMalloc(&hidden_state, 512 * sizeof(__nv_bfloat16) * 2048);
  cudaMemcpy(hidden_state, input_embeddings, 512 * sizeof(__nv_bfloat16) * 2048,
             cudaMemcpyDeviceToDevice);

  rmsNormKernel<<<num_input_tokens, 1024>>>(input_embeddings, hidden_state,
                                            NULL);

  // attention()
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::cerr << "CUDA launch failed: " << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
  }
//
//   for (size_t t = 0; t < 10; ++t) {
//     // std::cout << "Token " << input_tokens[t] << ": ";
//
//     // for (int j = 1024; j < 1030; ++j) {
//     //     size_t index = t * 2048 + j;
//     std::cout << __bfloat162float(hidden_state[t]) << ' ';
//     // }
//   }
}
