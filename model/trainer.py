"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from model.input import Input
from model.gcn import GCNClassifier
from utils.torch_utils import  get_long_tensor, set_cuda, change_lr, get_optimizer



class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None, ucca_embedding_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.ucca_embedding_matrix = ucca_embedding_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix, ucca_embedding_matrix=ucca_embedding_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        input, labels = self.unpack_batch(batch, self.opt['cuda'] )

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(input)
        loss = self.criterion(logits, labels)

        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']

        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()

        loss_val = loss.item()

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()

        return loss_val

    def predict(self, batch, unsort=True):
        input, labels = self.unpack_batch(batch, self.opt['cuda'] )

        #inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = input.orig_idx
        ids = input.id

        # forward
        self.model.eval()
        logits, _ = self.model(input)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs, ids = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs, ids)))]
        return predictions, probs, loss.item(), ids

    def unpack_batch(self, batch, cuda):

        words = set_cuda(get_long_tensor(batch.word, batch.batch_size), cuda)
        masks = set_cuda(torch.eq(words, 0), cuda)
        pos = set_cuda(get_long_tensor(batch.pos, batch.batch_size), cuda)
        ner = set_cuda(get_long_tensor(batch.ner, batch.batch_size), cuda)
        coref = set_cuda(get_long_tensor(batch.coref, batch.batch_size), cuda)
        ucca_enc = set_cuda(get_long_tensor(batch.ucca_enc, batch.batch_size), cuda)

        rel = set_cuda(torch.LongTensor(batch.rel), cuda)

        input = Input(batch_size=batch.batch_size,
                      word=words,
                      mask=masks,
                      pos=pos,
                      ner=ner,
                      coref=coref,
                      ucca_enc=ucca_enc,
                      len=batch.len,
                      head=batch.head,
                      ucca_head=batch.ucca_head,
                      ucca_multi_head=batch.ucca_multi_head,
                      ucca_dist_from_mh_path=batch.ucca_dist_from_mh_path,
                      subj_p=batch.subj_p,
                      obj_p=batch.obj_p,
                      id=batch.id,
                      orig_idx=batch.orig_idx)

        return input, rel



