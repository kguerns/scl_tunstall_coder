from scl.core.data_block import DataBlock

from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.utils.test_utils import try_lossless_compression, are_blocks_equal

from dataclasses import dataclass

import argparse
import os
import sys

# Directory of the current file (eval/eval.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNSTALL_DIR = os.path.join(CURRENT_DIR, "../scl/compressors")
sys.path.append(TUNSTALL_DIR)

from tunstall_coder import TunstallEncoder, TunstallSerialDecoder, TunstallParallelDecoder

DATA_BLOCK_SIZE = 50000

@dataclass
class DecodingResult:
    decoder_name: str
    decode_time: float

@dataclass
class Result:
    entropy: float
    avg_bits: float
    decode_results: list

def read_file_as_bytes(file_path, block_size=50000):
    """
    Read any file (text or binary) as raw bytes.
    """
    with open(file_path, 'rb') as f:  # 'rb' = read binary
        data = f.read(block_size)
    
    return data  # bytes object
    # Or convert to list: return list(data)

def read_file_as_chars(file_path, block_size=50000):
    """
    Read any file (text or binary) as a list of characters.
    Each byte is converted to a character via chr().
    """
    with open(file_path, 'rb') as f:  # 'rb' = read binary
        data = f.read(block_size)
    
    # Convert bytes to characters
    char_list = [chr(b) for b in data]
    
    return char_list

def compress_decompress(data_block, encoder, decoder):
    import time
    # print(f"data_block: {data_block.data_list[:100]}")
    encoded_bitarray = encoder.encode_block(data_block)

    # test decode
    start = time.perf_counter()
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
    end = time.perf_counter()
    decoding_time = end - start

    # print(f"decode_block: {decoded_block.data_list[:100]}")
    assert num_bits_consumed == len(encoded_bitarray), "Decoder did not consume all bits"

    # compare blocks
    return are_blocks_equal(data_block, decoded_block), len(encoded_bitarray), decoding_time


def test_tunstall_coder(file_path, code_length):
    # data = read_file_as_bytes(file_path, DATA_BLOCK_SIZE)
    # data_block = DataBlock(list(data))
    data_list = read_file_as_chars(file_path, DATA_BLOCK_SIZE)
    data_block = DataBlock(data_list)
    
    prob_dist = data_block.get_empirical_distribution()
    encoder = TunstallEncoder(prob_dist, code_length)

    decoder_names = ["Serial Decoder", "Parallel Decoder"]
    decoders = [TunstallSerialDecoder(prob_dist, code_length), TunstallParallelDecoder(prob_dist, code_length)]
    assert len(decoder_names) == len(decoders)

    decode_results = []
    for i in range(len(decoders)):

        # is_lossless, output_len, encoded_bitarray = try_lossless_compression(data_block, encoder, decoder)
        is_lossless, encoded_len, decode_time = compress_decompress(data_block, encoder, decoders[i])
        avg_bits = encoded_len / data_block.size

        assert is_lossless, f"Lossless compression failed with {decoder_names[i]}"

        decode_result = DecodingResult(
            decoder_name=decoder_names[i],
            decode_time=decode_time,
        )
        decode_results.append(decode_result)

    result = Result(
        entropy=prob_dist.entropy,
        avg_bits=avg_bits,
        decode_results=decode_results
    )
    return result


def main(input_folder, code_length):
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)

        result = test_tunstall_coder(file_path, code_length)
        str_to_print = f"""
{file}:
    Entropy: {result.entropy:>8.4f}, Avg Bits: {result.avg_bits:>8.4f}
    Decoding time:"""
        for decode_result in result.decode_results:
            str_to_print += f"""
        {decode_result.decoder_name:<20}: {decode_result.decode_time * 1000:>10.3f} ms"""
        print(str_to_print)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating performance of Tunstall code.")

    parser.add_argument("input_folder", help="Path to the folder containing all files")
    parser.add_argument("--code_length", type=int, default=10, help="Tunstall code legnth")

    args = parser.parse_args()

    main(args.input_folder, args.code_length)