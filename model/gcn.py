"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import itertools
import networkx

from model.tree import Tree, head_to_tree, tree_to_adj, fold_multiple_root_words
from utils import constant
from utils.torch_utils import  keep_partial_grad, get_long_tensor, set_cuda
from utils.ucca_embedding import UccaEmbedding


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None, ucca_embedding_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix, ucca_embedding_matrix=ucca_embedding_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None, ucca_embedding_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = torch.from_numpy(emb_matrix) if not emb_matrix is None else None
        self.ucca_embedding_matrix = torch.from_numpy(ucca_embedding_matrix) if not ucca_embedding_matrix is None else None

        # UD segment
        self.ud_emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.ud_pos_emb = nn.Embedding(len(constant.SPACY_POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ud_ner_emb = nn.Embedding(len(constant.SPACY_NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.ud_ucca_path_emb = nn.Embedding(opt['ucca_embedding_vocab_size'], opt['ucca_embedding_dim']) if opt['ucca_embedding_dim'] > 0 and opt['ucca_embedding_for_ud_too'] else None
        self.ud_coref_emb = nn.Embedding(len(constant.ALL_NER_TYPES)*3, opt['coref_dim']) if opt['coref_dim'] > 0 else None

        self.init_embeddings(self.ud_emb, self.ud_ucca_path_emb)
        ud_embeddings = (self.ud_emb, self.ud_pos_emb, self.ud_ner_emb, self.ud_ucca_path_emb, self.ud_coref_emb)
        self.gcn_ud = GCN(opt, ud_embeddings, opt['hidden_dim'], opt['num_layers'])

        # UCCA segment
        self.ucca_emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.ucca_pos_emb = nn.Embedding(len(constant.SPACY_POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ucca_ner_emb = nn.Embedding(len(constant.SPACY_NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.ucca_ucca_path_emb = nn.Embedding(opt['ucca_embedding_vocab_size'], opt['ucca_embedding_dim']) if opt['ucca_embedding_dim'] > 0 else None
        self.ucca_coref_emb = nn.Embedding(len(constant.ALL_NER_TYPES)*3, opt['coref_dim']) if opt['coref_dim'] > 0 else None

        self.init_embeddings(self.ucca_emb, self.ucca_ucca_path_emb)
        ucca_embeddings = (self.ucca_emb, self.ucca_pos_emb, self.ucca_ner_emb, self.ucca_ucca_path_emb, self.ucca_coref_emb)
        self.gcn_ucca = GCN(opt, ucca_embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers

        in_dim = opt['hidden_dim'] * (2 if opt['gcn_pooling_method'] == 'seperate_gcn' else 3)
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self, emb, ucca_path_emb):
        if self.emb_matrix is None:
            emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            emb.weight.data.copy_(self.emb_matrix)

        if not ucca_path_emb is None and not self.opt['ucca_embedding_ignore']:
            ucca_path_emb.weight.data.copy_(self.ucca_embedding_matrix)

        # decide finetuning (for word embeddings, not the other embeddings)
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            emb.weight.register_hook(lambda x: keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        maxlen = max(inputs.len)

        def trees_to_adj(heads, l, prune, subj_pos, obj_pos):
            trees = [head_to_tree(fold_multiple_root_words(heads[i]), prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]

            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)

            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        def dags_to_adj_from_dist_to_path(multi_heads, distances, l, prune):
            batch_len = len(multi_heads)

            adj_matrices = []
            for sent_idx in range(batch_len):
                sent_len = l[sent_idx]
                multi_head = multi_heads[sent_idx]
                distance = distances[sent_idx]
                adj_matrix = np.zeros((maxlen, maxlen), dtype=np.float32)

                for i in range(sent_len):
                    if distance[i] <= prune:
                        for head in multi_head[i]:
                            if head > 0 and distance[head-1] <= prune:
                                adj_matrix[head-1,i] = 1

                adj_matrix = adj_matrix + adj_matrix.T

                adj_matrices.append(adj_matrix.reshape(1, maxlen, maxlen))

            adj = np.concatenate(adj_matrices, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        def dags_to_adj(multi_heads, l, prune, subj_pos, obj_pos):
            batch_len = len(multi_heads)

            adj_matrices = []
            for sent_idx in range(batch_len):
                sent_len = l[sent_idx]
                adj_matrix = np.zeros((maxlen, maxlen), dtype=np.float32)
                subj = [token_id for token_id in range(sent_len) if subj_pos[sent_idx][token_id] == 0]
                obj= [token_id for token_id in range(sent_len) if obj_pos[sent_idx][token_id] == 0]
                edges = {(head_id-1, token_id):True for token_id in range(sent_len) for head_id in multi_heads[sent_idx][token_id] }

                extended_edges = edges.copy()
                for edge in itertools.combinations(subj, 2):
                    extended_edges[edge] = True
                for edge in itertools.combinations(obj, 2):
                    extended_edges[edge] = True

                graph = networkx.Graph(list(extended_edges.keys()))
                try:
                    on_path = networkx.shortest_path(graph, source=subj[0], target=obj[0])
                except networkx.NetworkXNoPath:
                    #adj_matrix = np.ones((maxlen, maxlen), dtype=np.float32)
                    print('shit')
                    adj_matrices.append(adj_matrix.reshape(1, maxlen, maxlen))
                    continue

                all_shortest_paths_lengths = {start: targets for start, targets in networkx.shortest_path_length(graph)}

                token_distances = []
                for token_id in range(sent_len):
                    distance = 0
                    if token_id not in on_path:
                        distances_to_path = {target: distance_to_target
                                             for target, distance_to_target in all_shortest_paths_lengths[token_id].items()
                                             if target in on_path}
                        distance = min(distances_to_path.values(), default=constant.INFINITY_NUMBER)

                    token_distances.append(distance)

                dgraph = networkx.DiGraph(list(edges.keys()))
                for i in dgraph.nodes():
                    if i >= 0 and token_distances[i] <= prune:
                        for j in dgraph.successors(i):
                            if j >= 0 and token_distances[j] <= prune:
                                adj_matrix[i, j] = 1
                adj_matrix = adj_matrix + adj_matrix.T

                adj_matrices.append(adj_matrix.reshape(1, maxlen, maxlen))

            adj = np.concatenate(adj_matrices, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        ud_adj = trees_to_adj(inputs.head, inputs.len, self.opt['prune_k'], inputs.subj_p, inputs.obj_p)
        ud_gcn_out, ud_gcn_out_mask = self.gcn_ud(ud_adj, inputs)
        ud_gcn_out = ud_gcn_out.masked_fill(ud_gcn_out_mask, -constant.INFINITY_NUMBER)

        ucca_adj = dags_to_adj_from_dist_to_path(inputs.ucca_multi_head, inputs.ucca_dist_from_mh_path, inputs.len, self.opt['prune_k'])
        ucca_gcn_out, ucca_gcn_out_mask = self.gcn_ucca(ucca_adj, inputs)
        ucca_gcn_out = ucca_gcn_out.masked_fill(ucca_gcn_out_mask, -constant.INFINITY_NUMBER)


      # pooling
        if self.opt['gcn_pooling_method'] == 'merged_gcn_plus_subj_and_obj':
            gcn_out = torch.max(ud_gcn_out, ucca_gcn_out)

            subj_mask = set_cuda(get_long_tensor(inputs.subj_p, inputs.batch_size ), self.opt['cuda']).eq(0).eq(0).unsqueeze(2) # invert mask
            obj_mask = set_cuda(get_long_tensor(inputs.obj_p, inputs.batch_size ), self.opt['cuda']).eq(0).eq(0).unsqueeze(2) # invert mask

            if self.opt['fix_subj_obj_mask_bug']:
                pool_mask = ud_gcn_out_mask & ucca_gcn_out_mask

                subj_mask = ~(~subj_mask & ~pool_mask)
                obj_mask = ~(~obj_mask & ~pool_mask)

            h_out = torch.max(gcn_out, 1)[0]
            subj_out = pool(gcn_out, subj_mask)
            obj_out = pool(gcn_out, obj_mask)

            outputs = torch.cat([h_out, subj_out, obj_out], dim=1)

        elif self.opt['gcn_pooling_method'] == 'seperate_gcn':
            outputs = torch.cat([ud_gcn_out, ucca_gcn_out], dim=1)
            h_out = torch.max(outputs, 1)[0]

            ud_gcn_out = torch.max(ud_gcn_out, 1)[0]
            ucca_gcn_out = torch.max(ucca_gcn_out, 1)[0]

            outputs = torch.cat([ud_gcn_out, ucca_gcn_out], dim=1)

        outputs = self.out_mlp(outputs)

        return outputs, h_out


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['ucca_embedding_dim'] + opt['coref_dim']

        self.emb, self.pos_emb, self.ner_emb, self.ucca_emb, self.coref_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'], use_cuda=self.use_cuda)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):

        word_embs = self.emb(inputs.word)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(inputs.pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(inputs.ner)]
        if self.opt['ucca_embedding_dim'] > 0:
            embs += [self.ucca_emb(inputs.ucca_enc)]
        if self.opt['coref_dim'] > 0:
            embs += [self.coref_emb(inputs.coref)]


        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, inputs.mask, inputs.word.size()[0]))
        else:
            gcn_inputs = embs
        
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)
            wait_here = True


        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        if self.opt['mask_in_self_loop']:
            flip_mask = adj.sum(2).gt(0).float().unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)

            if self.opt['mask_in_self_loop']:
                gcn_inputs_filtered = torch.einsum('bxs,bxy -> bxy', flip_mask, gcn_inputs)
                AxW = AxW + self.W[l](gcn_inputs_filtered) # self loop
            else:
                AxW = AxW + self.W[l](gcn_inputs) # self loop

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask

def pool(h, mask):
    h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
    return torch.max(h, 1)[0]

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

