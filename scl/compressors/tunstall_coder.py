from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_block import DataBlock
from scl.core.prob_dist import ProbabilityDist

from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint

class TunstallCodebook:
    def __init__(self, prob_dist: ProbabilityDist, code_length):
        self.prob_dist = prob_dist
        self.alphabet = prob_dist.alphabet
        self.code_length = code_length
        self.max_codebook_size = 2 ** code_length

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
        
        while len(self.codebook) + len(self.alphabet) - 1 <= self.max_codebook_size:
            # get the the phrase with the max prob
            max_item = max(self.codebook.items(), key=lambda x: x[1])
            phrase, prob = max_item

            del phrase_prob_dict[phrase]
            
            # pad the phrase with each symbol, 
            # and insert new_phrase with new_prob to phrase_prob_dict
            for symbol in self.alphabet:
                new_phrase = phrase + symbol
                new_prob = prob * self.prob_dist.probability(symbol)
                phrase_prob_dict[new_phrase] = new_prob
        
        # assign a codeword to each phrase in codebook
        for i, phrase in enumerate(phrase_prob_dict.keys()):
            codeword = uint_to_bitarray(i)
            self.codebook[phrase] = codeword


class TunstallEncoder(DataEncoder):

    def __init__(self, prob_dist: ProbabilityDist, code_length):
        tunstall_codebook = TunstallCodebook(prob_dist, code_length)
        self.codebook = tunstall_codebook.get_codebook()

    def encode_block(self, data_block: DataBlock):
        def shared_prefix_length(data_list, pos, phrase):
            count = 0
            while pos < len(data_list) and count < len(phrase):
                if data_list[pos+count] == phrase[count]:
                    count += 1
                else:
                    break
            return count
        
        encoded_bitarray = BitArray("")
        pos = 0
        while pos < data_block.size:
            # find longest matchlegnth
            chosen_phrase = ""
            for phrase in self.codebook:
                if shared_prefix_length(data_block.data_list, pos, phrase) > len(chosen_phrase):
                    chose_phrase = phrase
            
            # assert chosen_phrase != ""
            encoded_bitarray += self.codebook[chosen_phrase]
            pos += len(chosen_phrase)
        
        return encoded_bitarray
                

class TunstallDecoder(DataDecoder):
    """
    Tunstall code serial decoder
    """
    def __init__(self, prob_dist: ProbabilityDist, code_length):
        raise NotImplementedError
    
    def decode_block(self, bitarray: BitArray):
        raise NotImplementedError


class TunstallParallelDecoder(DataDecoder):
    """
    Tunstall code parallel decoder
    """

    def __init__(self, prob_dist: ProbabilityDist, code_length):
        raise NotImplementedError

    def decode_blcok(self, bitarray: BitArray):
        raise NotImplementedError