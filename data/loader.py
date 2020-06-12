"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils.ucca_embedding import UccaEmbedding
from utils import constant, helper, vocab
from collections import namedtuple

class Entry(namedtuple('Entry', 'token, pos, ner, head, multi_head, subj_p, obj_p, ucca_enc, ucca_dist_from_mh_path, rel, id')):
    """
    'Entry' objects represent individual TACRED entries, that have been preprocessed for further handling.
    """
    pass

class Batch(namedtuple('Batch', 'batch_size, word, pos, ner, head, multi_head, subj_p, obj_p, ucca_enc, ucca_dist_from_mh_path, rel, orig_idx, id, len')):
    """
    'Batch' objects hold batches of Entry objects (without no additional preprocessing)
    """
    pass


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, apply_filters=False, ucca_embedding=None):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.apply_filters = apply_filters
        self.label2id = constant.LABEL_TO_ID
        self.ucca_embedding = ucca_embedding
        self.field_to_index = {field:index for index, field in enumerate(Entry._fields)}

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        if not opt['train_without_shuffling'] and not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-2]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """

        processed = []

        for d in data:

            # if apply_filters set skip sentences with ucca_path_len that exceed max_ucca_path
            if self.apply_filters and opt['max_ucca_path'] > 0:
                ucca_path_len = d['ucca_path_len']
                # ignore sentence if either it's ucca_path_len is missing (i.e. equal to -1) or
                # if it's greater than positive opt['max_ucca_path']
                if ucca_path_len == -1 or ucca_path_len > opt['max_ucca_path']:
                    continue
            # if apply_filters set skip sentences with ud_path_len that exceed max_ud_path
            if self.apply_filters and opt['max_ud_path'] > 0:
                ud_path_len = d['ud_path_len']
                # ignore sentence if either it's ud_path_len is missing (i.e. equal to -1) or
                # if it's greater than positive opt['max_ud_path']
                if ud_path_len == -1 or ud_path_len > opt['max_ud_path']:
                    continue


            tac_to_ucca = { int(key):val for key, val in d['tac_to_ucca'].items() }
            tokens = list(d['ucca_tokens'])
            l = len(tokens)

            if opt['lower']:
                tokens = [t.lower() for t in tokens]

            d['subj_start'] = tac_to_ucca[d['subj_start']][0]
            d['subj_end'] = tac_to_ucca[d['subj_end']][-1]
            d['obj_start'] = tac_to_ucca[d['obj_start']][0]
            d['obj_end'] = tac_to_ucca[d['obj_end']][-1]

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)

            if opt['primary_engine'] == 'spacy':
                pos = map_to_ids(d['spacy_tag'], constant.SPACY_POS_TO_ID)
                ner = map_to_ids(d['spacy_ent'], constant.SPACY_NER_TO_ID)
            else:
                pos = map_to_ids(d['corenlp_pos'], constant.POS_TO_ID)
                ner = map_to_ids(d['corenlp_ner'], constant.NER_TO_ID)

            #deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)

            if opt['head'] == 'ucca':
                heads = [int(x) for x in d['ucca_heads']]

            elif opt['primary_engine'] == 'spacy':
                heads = [int(x) for x in d['spacy_head']]

            else:
                heads = [int(x) for x in d['corenlp_heads']]

            multi_heads = [[head for dep, head in ucca_deps] for ucca_deps in d['ucca_deps']]
            ucca_dist_from_mh_path = [int(x) for x in d['dist_from_ucca_mh_path']]

            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)

            relation = self.label2id[d['relation']]

            # capture UCCA encoding
            ucca_encodings_for_min_subtree = []
            if opt['ucca_embedding_dim'] > 0:

                ucca_embedding_source = 'ucca_encodings_min_subtree' if opt['ucca_embedding_source'] == 'min_sub_tree' else 'ucca_encodings'

                assert(ucca_embedding_source in d)

                index_to_encoding = {int(k):v for k,v in d[ucca_embedding_source].items()} if d[ucca_embedding_source] is not None else {}
                ucca_encodings_for_min_subtree = []

                for index in range(0, len(tokens)):
                    if index in index_to_encoding:
                        ucca_encodings_for_min_subtree.append(self.ucca_embedding.get_index(index_to_encoding[index]))
                    else:
                        ucca_encodings_for_min_subtree.append(self.ucca_embedding.get_index(''))


            if opt['primary_engine'] == 'corenlp' and opt['coref_dim'] > 0:
                assert('corenlp_coref' in d)

                subject_index_range = range(d['subj_start'], d['subj_end']+1)
                object_index_range = range(d['obj_start'], d['obj_end']+1)

                #"corenlp_coref": [[[25, 26], [[35, 36]]]]

                coref = ['O'] * len(tokens)

                for coref_set in d['corenlp_coref']:
                    anchor_coords = coref_set[0]
                    anchor = set(index-1 for index in range(anchor_coords[0],anchor_coords[1]))
                    anchor_ner = next((ner[index] for index in anchor if ner[index] != 'O'), 'O')
                    if anchor_ner == 'O':
                        continue

                    coref_to_what = 'NO_M'
                    if anchor.intersection(subject_index_range):
                        coref_to_what = 'M_SUBJ'
                    elif anchor.intersection(object_index_range):
                        coref_to_what = 'M_OBJ'

                    ref_coord_pairs = coref_set[1]
                    for ref_coord_pair in ref_coord_pairs:
                        for i in [index-1 for index in range(ref_coord_pair[0],ref_coord_pair[1])]:
                            coref[i] = constant.COREF_TO_ID[coref_to_what][anchor_ner]



            # capture id so that we can propagate through model
            tacred_id = d['id']

            data_entry = Entry(token=tokens,
                               pos=pos,
                               ner=ner,
                               head=heads,
                               multi_head=multi_heads,
                               subj_p=subj_positions,
                               obj_p=obj_positions,
                               ucca_enc=ucca_encodings_for_min_subtree,
                               ucca_dist_from_mh_path=ucca_dist_from_mh_path,
                               rel=relation,
                               id=tacred_id)

            processed.append(data_entry)

        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)

        #transpose
        batch = list(zip(*batch))
        assert len(batch) == len(self.field_to_index)


        tokens = batch[self.field_to_index['token']]

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in tokens]
        batch, orig_idx = sort_all(batch, lens)

        tokens = batch[self.field_to_index['token']]


        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in tokens]
        else:
            words = tokens


        pos = batch[self.field_to_index['pos']]
        ner = batch[self.field_to_index['ner']]
        heads = batch[self.field_to_index['head']]
        multi_heads = batch[self.field_to_index['multi_head']]
        subj_p = batch[self.field_to_index['subj_p']]
        obj_p = batch[self.field_to_index['obj_p']]
        ucca_enc = batch[self.field_to_index['ucca_enc']]
        ucca_dist_from_mh_path = batch[self.field_to_index['ucca_dist_from_mh_path']]
        rel = batch[self.field_to_index['rel']]
        id = batch[self.field_to_index['id']]


        return Batch(batch_size=batch_size,
                     word=words,
                     pos=pos,
                     ner=ner,
                     head=heads,
                     multi_head=multi_heads,
                     subj_p=subj_p,
                     obj_p=obj_p,
                     ucca_enc=ucca_enc,
                     ucca_dist_from_mh_path=ucca_dist_from_mh_path,
                     rel=rel,
                     orig_idx=orig_idx,
                     id=id,
                     len=sorted(lens, reverse=True))


    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

