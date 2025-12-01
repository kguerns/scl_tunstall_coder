from scl.core.data_block import DataBlock

from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.utils.test_utils import try_lossless_compression, are_blocks_equal

from dataclasses import dataclass

import argparse
import os
import sys
import time

# Directory of the current file (eval/eval.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TUNSTALL_DIR = os.path.join(CURRENT_DIR, "../scl/compressors")
sys.path.append(TUNSTALL_DIR)

from tunstall_coder import TunstallEncoder, TunstallSerialDecoder, TunstallParallelDecoder
from tunstall_cuda_decoder import TunstallParallelCudaDecoder

DATA_BLOCK_SIZE = 1024 * 512 # 512 KB

@dataclass
class DecodingResult:
    decoder_name: str
    decode_time: float

@dataclass
class Result:
    entropy: float
    avg_bits: float
    compression_ratio: float
    encoding_time: float
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
    start = time.perf_counter()
    encoded_bitarray = encoder.encode_block(data_block)
    end = time.perf_counter()
    encoding_time = end - start
    print(f"Encoding time: {encoding_time * 1000:>10.3f} ms")

    # test decode
    start = time.perf_counter()
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
    end = time.perf_counter()
    decoding_time = end - start

    # print(f"decode_block: {decoded_block.data_list[:100]}")
    assert num_bits_consumed == len(encoded_bitarray), "Decoder did not consume all bits"

    # compare blocks
    return are_blocks_equal(data_block, decoded_block), len(encoded_bitarray), decoding_time


def decode_only(data_block, encoded_bitarray, decoder):
    import time

    # test decode
    start = time.perf_counter()
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
    end = time.perf_counter()
    decoding_time = end - start

    assert num_bits_consumed == len(encoded_bitarray), "Decoder did not consume all bits"

    # compare blocks
    return are_blocks_equal(data_block, decoded_block), decoding_time


def test_tunstall_coder(file_path, code_length):
    import time
    # data = read_file_as_bytes(file_path, DATA_BLOCK_SIZE)
    # data_block = DataBlock(list(data))
    data_list = read_file_as_chars(file_path, DATA_BLOCK_SIZE)
    data_block = DataBlock(data_list)

    prob_dist = data_block.get_empirical_distribution()
    encoder = TunstallEncoder(prob_dist, code_length)

    # Encode once
    start = time.perf_counter()
    encoded_bitarray = encoder.encode_block(data_block)
    end = time.perf_counter()
    encoding_time = end - start
    print(f"Encoding time: {encoding_time * 1000:>10.3f} ms")

    avg_bits = len(encoded_bitarray) / data_block.size

    # 8 bits per symbol / avg bits per symbol
    compression_ratio = 8 / avg_bits

    decoder_names = ["Serial Decoder", "Parallel Decoder", "Parallel CUDA Decoder"]
    decoders = [
        TunstallSerialDecoder(prob_dist, code_length),
        TunstallParallelDecoder(prob_dist, code_length),
        TunstallParallelCudaDecoder(prob_dist, code_length)
    ]
    assert len(decoder_names) == len(decoders)

    decode_results = []
    for i in range(len(decoders)):
        # Decode using pre-encoded bitarray
        is_lossless, decode_time = decode_only(data_block, encoded_bitarray, decoders[i])

        assert is_lossless, f"Lossless compression failed with {decoder_names[i]}"

        decode_result = DecodingResult(
            decoder_name=decoder_names[i],
            decode_time=decode_time,
        )
        decode_results.append(decode_result)

    result = Result(
        entropy=prob_dist.entropy,
        avg_bits=avg_bits,
        compression_ratio=compression_ratio,
        encoding_time=encoding_time,
        decode_results=decode_results
    )
    return result


def main(input_folder, code_length):
    def format_size(bytes):
      """Convert bytes to human readable format"""
      for unit in ['B', 'KB', 'MB', 'GB']:
          if bytes < 1024.0:
              return f"{bytes:.2f} {unit}"
          bytes /= 1024.0
      return f"{bytes:.2f} TB"
    
    all_results = []
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        file_size = os.path.getsize(file_path)

        print(f"\nWorking on {file}: {format_size(file_size)}")

        result = test_tunstall_coder(file_path, code_length)
        str_to_print = f"""{file}: {format_size(file_size)}
    Entropy: {result.entropy:>8.4f}, Avg Bits: {result.avg_bits:>8.4f}
    Decoding time:"""
        for decode_result in result.decode_results:
            str_to_print += f"""
        {decode_result.decoder_name:<20}: {decode_result.decode_time * 1000:>10.3f} ms"""
        print(str_to_print)
        all_results.append((file, result))

    # save all results to a file
    output_filename = f"results_codelength_{code_length}.txt"

    with open(output_filename, 'w') as f:
        f.write(f"# Tunstall Coding Evaluation Summary (Code Length: {code_length})\n")
        f.write(f"# Generated: {time.ctime()}\n\n")

        for file_name, result in all_results:
            f.write(f"--- File: {file_name} ---\n")
            f.write(f"Entropy: {result.entropy:.4f}\n")
            f.write(f"Avg Bits/Sym: {result.avg_bits:.4f}\n")
            f.write(f"Compression Ratio: {result.compression_ratio}\n")
            f.write(f"Encoding Time (ms): {result.encoding_time* 1000:>10.3f}\n")

            f.write("Decoding Time (ms):\n")
            for decode_result in result.decode_results:
                f.write(f"\t{decode_result.decoder_name:<20}: {decode_result.decode_time * 1000:>10.3f}\n")
            f.write("\n")

    print(f"Wrote all results to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating performance of Tunstall code.")

    parser.add_argument("input_folder", help="Path to the folder containing all files")
    parser.add_argument("--code_length", type=int, default=10, help="Tunstall code legnth")

    args = parser.parse_args()

    main(args.input_folder, args.code_length)