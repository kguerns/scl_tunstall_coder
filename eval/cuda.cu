#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define THREADS_PER_BLOCK 256

__global__ void bits_to_indices_kernel(
    const uint8_t* bits,
    const int* phrase_lengths,
    int* codeword_indices,
    int* output_phrase_lengths,
    const int num_codewords,
    const int code_length
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_codewords) {
        return;
    }

    const int bit_start_idx = code_length * idx;

    int codeword_int = 0;
    for (int i = 0; i < code_length; i++) {
        codeword_int = (codeword_int << 1) | (int)bits[bit_start_idx + i];
    }

    codeword_indices[idx] = codeword_int;
    output_phrase_lengths[idx] = phrase_lengths[codeword_int];
}

__global__ void decode_kernel(
    const int* codeword_indices,
    const uint8_t* phrase_chars,
    const int* phrase_offsets,
    const int* phrase_lengths,
    uint8_t* output,
    const int* output_offsets,
    const int num_codewords
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_codewords) {
        return;
    }

    int codeword_int = codeword_indices[idx];

    int phrase_start = phrase_offsets[codeword_int];
    int phrase_len = phrase_lengths[codeword_int];

    int output_offset = output_offsets[idx];

    for (int i = 0; i < phrase_len; i++) {
        output[output_offset + i] = phrase_chars[phrase_start + i];
    }
}

torch::Tensor decode(
    torch::Tensor bits,
    torch::Tensor phrase_chars,
    torch::Tensor phrase_offsets,
    torch::Tensor phrase_lengths,
    int code_length
) {
    const int num_bits = bits.size(0);
    const int num_codewords = num_bits / code_length;

    int blocks = (num_codewords + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto codeword_indices = torch::zeros({num_codewords}, torch::dtype(torch::kInt32).device(bits.device()));
    auto codeword_phrase_lengths = torch::zeros({num_codewords}, torch::dtype(torch::kInt32).device(bits.device()));

    bits_to_indices_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        bits.data_ptr<uint8_t>(),
        phrase_lengths.data_ptr<int>(),
        codeword_indices.data_ptr<int>(),
        codeword_phrase_lengths.data_ptr<int>(),
        num_codewords,
        code_length
    );

    auto output_offsets = torch::zeros({num_codewords}, torch::dtype(torch::kInt32).device(bits.device()));
    thrust::device_ptr<int> phrase_lengths_ptr(codeword_phrase_lengths.data_ptr<int>());
    thrust::device_ptr<int> output_offsets_ptr(output_offsets.data_ptr<int>());
    thrust::exclusive_scan(phrase_lengths_ptr, phrase_lengths_ptr + num_codewords, output_offsets_ptr);

    auto last_offset = output_offsets[-1].item<int>();
    auto last_length = codeword_phrase_lengths[-1].item<int>();
    int total_output_size = last_offset + last_length;

    auto output = torch::zeros({total_output_size}, torch::dtype(torch::kUInt8).device(bits.device()));

    decode_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        codeword_indices.data_ptr<int>(),
        phrase_chars.data_ptr<uint8_t>(),
        phrase_offsets.data_ptr<int>(),
        phrase_lengths.data_ptr<int>(),
        output.data_ptr<uint8_t>(),
        output_offsets.data_ptr<int>(),
        num_codewords
    );

    return output;
}
