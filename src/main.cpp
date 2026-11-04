#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <driver_types.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0 // this one is useful, but I don't remember why
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include "kernels.h"

using json = nlohmann::json;

int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem << "GB, total memory: " << total_mem << "GB\n";
    return 0;
}

int main() {
    checkGPUStatus();

    // Open the text file for reading
    std::ifstream safetensors_file("models/model.safetensors", std::ios_base::binary);
    std::ifstream tokenizer_file("models/tokenizer.json", std::ios_base::binary);

    // if (!safetensors_file) {
    //     std::cerr << "Error opening file for reading.";
    //     return 1;
    // }

    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
    std::string header_json(header_size, '\0');
    safetensors_file.read(header_json.data(), header_size);

    const auto& parsed_json = json::parse(header_json);

    const auto& tensor = parsed_json.at("model.embed_tokens.weight");

    std::string dtype = tensor.at("dtype").get<std::string>();
    uint64_t offset = tensor.at("data_offsets").at(1).get<uint64_t>();

    std::cout << tensor.dump(2) << '\n';

    // uint16_t table[262668288];
    // safetensors_file.read(reinterpret_cast<char *>(&table), offset);
    //
    static_assert(sizeof(__nv_bfloat16) == 2);
    std::vector<__nv_bfloat16> table(offset / sizeof(__nv_bfloat16));

    safetensors_file.seekg(8 + header_size);

    if (!safetensors_file.read(
            reinterpret_cast<char*>(table.data()), offset)) {
        std::cerr << "Failed to read embedding tensor\n";
        return 1;
    }

    std::cout << __bfloat162float(table[0]) << '\n';

    // All tensors are now loaded on CPU

    std::vector<int> input_tokens = {678, 264, 1933, 13};

    constexpr int MAX_PROMPT_LEN = 512; // max input tokens count

    int *gpu_input_tokens;

    cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
    cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);

    __nv_bfloat16* input_embeddings;
    cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * 2048);

    __nv_bfloat16* table_embeddings_gpu;
    cudaMalloc(&table_embeddings_gpu, table.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(table_embeddings_gpu, table.data(), table.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    embeddingGather(gpu_input_tokens, input_embeddings, table_embeddings_gpu, input_tokens.size());

    std::vector<__nv_bfloat16> cpu_input_embeddings(input_tokens.size() * 2048);
    cudaMemcpy(cpu_input_embeddings.data(), input_embeddings, input_tokens.size() * sizeof(__nv_bfloat16) * 2048, cudaMemcpyDeviceToHost);
    for (size_t t = 0; t < input_tokens.size(); ++t) {
        std::cout << "Token " << input_tokens[t] << ": ";

        for (int j = 0; j < 8; ++j) {
            size_t index = t * 2048 + j;
            std::cout << __bfloat162float(cpu_input_embeddings[index]) << ' ';
        }

        std::cout << "... \n";

        // CPU ONLY COMPARISON
        std::cout << "CPU Token " << input_tokens[t] << ": ";

        for (int j = 0; j < 8; ++j) {
            size_t index = input_tokens[t] * 2048 + j;
            std::cout << __bfloat162float(table[index]) << ' ';
        }

        std::cout << "... \n";
    }

        // Read each line from the filei
   // while (std::getline(f, s))
    //    std::cout << s << std::endl;

    // Close the file
    safetensors_file.close();
    // free(cpu_input_embeddings);

    return 0;
}
