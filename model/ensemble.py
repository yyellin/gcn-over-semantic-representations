"""
ensemble evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np

from model.input import Input
from model.gcn import GCNClassifier

from utils import torch_utils, scorer, constant, helper



class GCNEnsembleEvaluator(object):

    def __init__(self, model_files):

        self.models = []

        for model_file in model_files:
            opt = torch_utils.load_config(model_file)
            model = GCNClassifier(opt)
            checkpoint = self.get_checkpoint(model_file)
            model.load_state_dict(checkpoint['model'])

            if opt['cuda']:
                model.cuda()

            self.models.append(model)


    def get_checkpoint(self, filename):
        try:
            checkpoint = torch.load(filename)
            return checkpoint
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()


    def predict(self, batch_per_model, cuda, unsort=True):

        anchor_input = None
        total_logits = None

        for model,batch in zip(self.models, batch_per_model):
            input, labels = Input.unpack_batch(batch, cuda)

            if anchor_input is None:
                anchor_input = input

            else:
                assert input.id == anchor_input.id
                assert input.orig_idx == anchor_input.orig_idx

            model.eval()
            logits, _ = model(input)

            if total_logits is None:
                total_logits = logits
            else:
                total_logits += logits


        probs = F.softmax(total_logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(total_logits.data.cpu().numpy(), axis=1).tolist()



        if unsort:
            _, predictions, probs, ids = [list(t) for t in zip(*sorted(zip(anchor_input.orig_idx, predictions, probs, anchor_input.id)))]

        return predictions, probs, ids




