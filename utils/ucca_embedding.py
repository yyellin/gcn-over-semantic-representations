"""
Class for UCCA Embedding operations

  Authors: Jonathan Yellin
  Status: prototype

"""

import itertools
from collections import Counter
import numpy as np
import pickle
import json
from utils import constant

class UccaEmbedding(object):

    def __init__(self, embedding_dim, index_file, embedding_file):
        with open(index_file, 'rb') as index:
            self.encoding_vocab = pickle.load(index)
        self.embedding_matrix = np.load(embedding_file)

        self.__encoding_to_index = {encoding:i for i, encoding in enumerate(self.encoding_vocab)}

    def get_index(self, encoding):
        return self.__encoding_to_index[encoding]


    @staticmethod
    def prepare(embedding_dim, input_files, index_file, embedding_file, source):

        encodings = itertools.chain(*[UccaEmbedding.__load_encodings(input_file, source) for input_file in input_files])
        counter = Counter(t for t in encodings)
        encoding_vocab = sorted([t for t in counter], key=counter.get, reverse=True)
        vocab_size = len(encoding_vocab)
        emb = np.zeros((vocab_size, embedding_dim))

        for i, embedding in enumerate(encoding_vocab):
            array_of_bits = []

            for character in embedding:
                bit_encoding = [1 if digit == '1' else 0 for digit in bin(constant.UCCA_DEP_TO_ID[character])[2:]]
                bit_encoding = [0] * (4-len(bit_encoding)) + bit_encoding
                assert(len(bit_encoding) == 4)

                array_of_bits += bit_encoding

            emb[i][0:len(array_of_bits)] = array_of_bits


        print("dumping to files...")
        with open(index_file, 'wb') as outfile:
            pickle.dump(encoding_vocab, outfile)

        with open(embedding_file, 'wb') as outfile:
            np.save(outfile, emb)

        print("all done.")

    @staticmethod
    def __load_encodings(filename, source):
        all_encodings = []

        with open(filename) as infile:

            data = json.load(infile)
            for d in data:
                num_tokens = len(d['ucca_tokens'])
                encodings_dict = d['ucca_encodings_min_subtree'] if source == 'min_sub_tree' else d['ucca_encodings']
                index_to_encoding = {int(k): v for k, v in encodings_dict.items()} if encodings_dict is not None else {}
                encodings = []

                for index in range(0, num_tokens):
                    if index in index_to_encoding:
                        encodings.append(index_to_encoding[index])
                    else:
                        encodings.append('')

                if not encodings_dict is None:
                    all_encodings += encodings

        print("{} encodings from {} samples loaded from {}.".format(len(all_encodings), len(data), filename))
        return all_encodings

