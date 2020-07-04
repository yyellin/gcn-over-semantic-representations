"""
ensemble evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np

from model.input import Input
from model.gcn import GCNClassifier





class GCNEnsembleEvaluator(object):

    def __init__(self, opt1, model1_file, opt2, model2_file):
        self.opt1 = opt1
        self.model1 = GCNClassifier(self.opt1)
        checkpoint1 = self.get_checkpoint(model1_file)
        self.model1.load_state_dict(checkpoint1['model'])

        self.opt2 = opt2
        self.model2 = GCNClassifier(self.opt2)
        checkpoint2 = self.get_checkpoint(model2_file)
        self.model2.load_state_dict(checkpoint2['model'])

        if opt1['cuda']:
            self.model1.cuda()

        if opt2['cuda']:
            self.model2.cuda()


    def get_checkpoint(self, filename):
        try:
            checkpoint = torch.load(filename)
            return checkpoint
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()


    def predict(self, batch1, batch2, cuda, unsort=True):

        input1, labels1 = Input.unpack_batch(batch1, cuda)
        self.model1.eval()
        logits1, _ = self.model1(input1)

        input2, labels2 = Input.unpack_batch(batch2, cuda)
        self.model2.eval()
        logits2, _ = self.model2(input2)

        logits = logits1 + logits2

        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        if unsort:
            _, predictions, probs, ids = [list(t) for t in zip(*sorted(zip(input.orig_idx, predictions, probs, input.id)))]

        return predictions, probs, ids




