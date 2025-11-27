import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNSTALL_DIR = os.path.join(CURRENT_DIR, "../scl/compressors")
sys.path.append(TUNSTALL_DIR)

from tunstall_coder import TunstallBaseDecoder
from scl.core.data_block import DataBlock
from scl.core.prob_dist import ProbabilityDist, get_avg_neg_log_prob
from scl.utils.bitarray_utils import BitArray

import torch
import numpy as np


def load_cuda_module():
    from torch.utils.cpp_extension import load_inline

    cuda_source_path = os.path.join(CURRENT_DIR, "cuda.cu")
    with open(cuda_source_path, 'r') as f:
        cuda_source = f.read()

    cpp_source = """
#include <torch/extension.h>

torch::Tensor decode(
    torch::Tensor bits,
    torch::Tensor phrase_chars,
    torch::Tensor phrase_offsets,
    torch::Tensor phrase_lengths,
    int code_length
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode, "Decode bits to characters (CUDA)");
}
"""

    cuda_module = load_inline(
        name='tunstall_cuda',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['decode'],
        verbose=True,
        with_cuda=True
    )

    return cuda_module


class TunstallParallelCudaDecoder(TunstallBaseDecoder):

    def __init__(self, prob_dist: ProbabilityDist, code_length, cuda_module=None):
        super().__init__(prob_dist, code_length)

        if cuda_module is None:
            self.cuda_module = load_cuda_module()
        else:
            self.cuda_module = cuda_module

        self._prepare_phrase_tensors()

    def _prepare_phrase_tensors(self):
        phrase_chars_list = []
        phrase_offsets = []
        phrase_lengths = []

        current_offset = 0
        for phrase in self.codeword_phrase_list:
            phrase_offsets.append(current_offset)
            phrase_lengths.append(len(phrase))

            for char in phrase:
                phrase_chars_list.append(ord(char))

            current_offset += len(phrase)

        self.phrase_chars_tensor = torch.tensor(phrase_chars_list, dtype=torch.uint8, device='cuda')
        self.phrase_offsets_tensor = torch.tensor(phrase_offsets, dtype=torch.int32, device='cuda')
        self.phrase_lengths_tensor = torch.tensor(phrase_lengths, dtype=torch.int32, device='cuda')

    def decode_block(self, bitarray: BitArray) -> DataBlock:
        bit_list = bitarray.tolist()
        assert len(bit_list) % self.code_length == 0

        bits_tensor = torch.tensor(bit_list, dtype=torch.uint8, device='cuda')

        decoded_chars_tensor = self.cuda_module.decode(
            bits_tensor,
            self.phrase_chars_tensor,
            self.phrase_offsets_tensor,
            self.phrase_lengths_tensor,
            self.code_length
        )

        decoded_chars = decoded_chars_tensor.cpu().numpy()
        decoded_data_list = [chr(c) for c in decoded_chars]

        decoded_block = DataBlock(decoded_data_list)
        return decoded_block, len(bit_list)
