#include <cuda_bf16.h>


void embeddingGather(
    int *gpu_input_tokens, // {32,34,12,...}: list of input token ids
    __nv_bfloat16 *out_input_embeddings, // output: N 2048-length vectors.
                                         // Embeddings for the input tokens
    __nv_bfloat16 *embed_tokens,         // all 128k embeddings
    int num_input_tokens
);
