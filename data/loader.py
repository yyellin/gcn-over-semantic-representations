"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, apply_filters=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.apply_filters = apply_filters
        self.label2id = constant.LABEL_TO_ID

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
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            # capture UCCA tokens on path
            if opt['use_ucca_words_on_path']:
                assert('ucca_words_on_path' in d)
                ucca_words_on_path = d['ucca_words_on_path'] if d['ucca_words_on_path']  is not None else []
                path_mask = [1 if i in ucca_words_on_path else 0 for i in range(0,l)]
            else:
                path_mask = [0] * l
            # capture id so that we can propagate through model
            tacred_id = d['id']
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, path_mask, relation, tacred_id)]
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
        batch = list(zip(*batch))
        assert len(batch) == 12

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)
        path_masks = get_float_tensor(batch[9], batch_size)
        rels = torch.LongTensor(batch[10])

        # don't forget to propogate TACRED ids ..
        tacred_ids = batch[11]

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, path_masks, rels, orig_idx, tacred_ids)

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

def get_long_tensor(tokens_list, batch_size, filler=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(filler)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size, filler=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(filler)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

