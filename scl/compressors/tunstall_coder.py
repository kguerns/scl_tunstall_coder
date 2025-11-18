from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_block import DataBlock
from scl.core.prob_dist import ProbabilityDist, get_avg_neg_log_prob
from scl.core.data_stream import TextFileDataStream

from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.utils.test_utils import try_lossless_compression, are_blocks_equal

import numpy as np
import argparse

class TunstallCodebook:
    def __init__(self, prob_dist: ProbabilityDist, code_length):
        self.prob_dist = prob_dist
        self.alphabet = prob_dist.alphabet
        self.code_length = code_length
        self.max_codebook_size = 2 ** code_length

        # We'll still have each symbol in the alphabet
        # in the codebook individually.
        assert self.max_codebook_size >= len(prob_dist.alphabet)

        self.codebook = {}
        self._build_tunstall_codebook()
    
    def get_codebook(self):
        return self.codebook
    
    def _build_tunstall_codebook(self):
        """
        From https://en.wikipedia.org/wiki/Tunstall_coding

        D := tree of |U| leaves, one for each letter of U.
        while |D| < max_codebook_size:
            Convert most probable leaf to tree with |U| leaves.
        """

        # initialize codebook with all symbols (phrases of length 1)
        phrase_prob_dict = {symbol: self.prob_dist.probability(symbol) for symbol in self.alphabet}

        # We want to include all single letters in the end
        # Otherwise, we might get not find a codeword for a remianing letter.
        missing_one_count = sum(1 for s in self.alphabet if s not in self.codebook)
        
        while len(phrase_prob_dict) + len(self.alphabet) - 1 + missing_one_count <= self.max_codebook_size:
            # get the the phrase with the max prob
            max_item = max(phrase_prob_dict.items(), key=lambda x: x[1])
            phrase, prob = max_item

            del phrase_prob_dict[phrase]            

            # pad the phrase with each symbol, 
            # and insert new_phrase with new_prob to phrase_prob_dict
            for symbol in self.alphabet:
                new_phrase = phrase + symbol
                new_prob = prob * self.prob_dist.probability(symbol)
                phrase_prob_dict[new_phrase] = new_prob
            
            missing_one_count = sum(1 for s in self.alphabet if s not in self.codebook)
        
        missing_symbols = [s for s in self.alphabet if s not in self.codebook]
        for symbol in missing_symbols:
            phrase_prob_dict[symbol] = self.prob_dist.probability(symbol)

        # sort the dictionary by the length of the keys
        phrase_prob_dict = dict(
            sorted(phrase_prob_dict.items(), key=lambda item: (-len(item[0]), item[0]))
        )

        # assign a codeword to each phrase in codebook
        for i, phrase in enumerate(phrase_prob_dict.keys()):
            codeword = uint_to_bitarray(i, bit_width=self.code_length)
            self.codebook[phrase] = codeword


class TunstallEncoder(DataEncoder):

    def __init__(self, prob_dist: ProbabilityDist, code_length):
        tunstall_codebook = TunstallCodebook(prob_dist, code_length)
        self.codebook = tunstall_codebook.get_codebook()

    def encode_block(self, data_block: DataBlock):
        def match(data_list, pos, phrase):
            if pos + len(phrase) > len(data_list):
                return False
            for i in range(len(phrase)):
                if data_list[pos+i] != phrase[i]:
                    return False
            return True

        encoded_bitarray = BitArray("")
        pos = 0
        while pos < data_block.size:
            # find longest matchlegnth
            chosen_phrase = ""
            for phrase in self.codebook:
                if match(data_block.data_list, pos, phrase):
                    chosen_phrase = phrase
                    break

            # assert chosen_phrase != ""
            encoded_bitarray += self.codebook[chosen_phrase]
            pos += len(chosen_phrase)

        return encoded_bitarray


class TunstallBaseDecoder(DataDecoder):
    def __init__(self, prob_dist: ProbabilityDist, code_length):
        self.code_length = code_length

        tunstall_codebook = TunstallCodebook(prob_dist, code_length)
        self.codebook = tunstall_codebook.get_codebook()

        self.codeword_phrase_list = self._get_codeword_phrase_list()

    def _get_codeword_phrase_list(self) -> np.array:
        codeword_phrase_dict = {}
        for phrase, codeword in self.codebook.items():
            integer = bitarray_to_uint(codeword)
            codeword_phrase_dict[integer] = phrase

        codeword_phrase_list = []
        for i in range(len(self.codebook)):
            codeword_phrase_list.append(codeword_phrase_dict[i])

        return np.array(codeword_phrase_list, dtype=object)


class TunstallSerialDecoder(TunstallBaseDecoder):
    """
    Tunstall code serial decoder
    """
    def __init__(self, prob_dist: ProbabilityDist, code_length):
        super().__init__(prob_dist, code_length)
    
    def decode_block(self, bitarray: BitArray) -> DataBlock:
        # raise NotImplementedError
        assert len(bitarray) % self.code_length == 0
        phrases_list = []
        for i in range(0, len(bitarray), self.code_length):
            codeword_int = bitarray_to_uint(bitarray[i:i+self.code_length])
            phrases_list.append(self.codeword_phrase_list[codeword_int])

        decoded_data_list = list(''.join(phrases_list))

        return DataBlock(decoded_data_list), len(bitarray)


class TunstallParallelDecoder(TunstallBaseDecoder):
    """
    Tunstall code parallel decoder
    """

    def __init__(self, prob_dist: ProbabilityDist, code_length):
        super().__init__(prob_dist, code_length)

    def decode_block(self, bitarray: BitArray) -> DataBlock:
        bit_list = bitarray.tolist()
        assert len(bit_list) % self.code_length == 0

        num_chunks = len(bit_list) // self.code_length

        bitarray_np = np.array(bit_list)

        codewords_np = bitarray_np.reshape(num_chunks, self.code_length)

        powers = 2 ** np.arange(self.code_length-1, -1, -1)
        indices = np.dot(codewords_np, powers)
    
        phrases = self.codeword_phrase_list[indices]

        phrases_list = phrases.tolist()
        decoded_data_list = list(''.join(phrases_list))

        decoded_block = DataBlock(decoded_data_list)
        return decoded_block, len(bit_list)


def compress_decompress(data_block, encoder, decoder):
    import time

    encoded_bitarray = encoder.encode_block(data_block)

    # test decode
    start = time.perf_counter()
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
    end = time.perf_counter()
    decoding_time = end - start
    assert num_bits_consumed == len(encoded_bitarray), "Decoder did not consume all bits"

    # compare blocks
    return are_blocks_equal(data_block, decoded_block), num_bits_consumed, decoding_time
    

def test_tunstall_coder(file_path, code_length):
    DATA_BLOCK_SIZE = 50000

    # read in DATA_BLOCK_SIZE bytes
    with TextFileDataStream(file_path, "r") as fds:
        data_block = fds.get_block(block_size=DATA_BLOCK_SIZE)
    
    prob_dist = data_block.get_empirical_distribution()
    encoder = TunstallEncoder(prob_dist, code_length)

    decoder_names = ["Serial Decoder", "Parallel Decoder"]
    decoders = [TunstallSerialDecoder(prob_dist, code_length), TunstallParallelDecoder(prob_dist, code_length)]
    assert len(decoder_names) == len(decoders)

    for i in range(len(decoders)):

        # is_lossless, output_len, encoded_bitarray = try_lossless_compression(data_block, encoder, decoder)
        is_lossless, output_len, decoding_time = compress_decompress(data_block, encoder, decoders[i])
        avg_bits = output_len / DATA_BLOCK_SIZE

        # get optimal codelen
        optimal_codelen = get_avg_neg_log_prob(prob_dist, data_block)
        assert is_lossless, "Lossless compression failed"
        
        print(
            f"{decoder_names[i]:<20}: "
            f"Avg Bits: {avg_bits:>8.4f}, "
            f"Entropy: {prob_dist.entropy:>8.4f}, "
            f"Decoding Time: {decoding_time * 1000:>10.3f} ms"
        )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing arguments for Tunstall code.")

    parser.add_argument("input_txt", help="Path to the input txt file")
    parser.add_argument("--code_length", type=int, default=10, help="Tunstall code legnth")

    args = parser.parse_args()

    test_tunstall_coder(args.input_txt, args.code_length)