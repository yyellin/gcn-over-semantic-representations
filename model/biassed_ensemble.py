"""
biased ensemble evaluation
"""


from collections import OrderedDict
from dataclasses import dataclass
from typing import List


import torch
import torch.nn.functional as F
import numpy as np

from model.input import Input
from model.gcn import GCNClassifier

from utils import torch_utils

@dataclass
class ModelStuff:
    representation: str
    dirs: List[str]
    files: List[str]
    data: object


class GCNBiassedEnsembleEvaluator(object):

    def __init__(self, model_stuff_list: List[ModelStuff]):

        self.models = OrderedDict()

        for model_stuff in model_stuff_list:

            self.models[model_stuff.representation] = []

            for model_file in model_stuff.files:
                opt = torch_utils.load_config(model_file)
                model = GCNClassifier(opt)
                checkpoint = self.get_checkpoint(model_file)
                model.load_state_dict(checkpoint['model'])

                if opt['cuda']:
                    model.cuda()

                self.models[model_stuff.representation].append(model)



    def get_checkpoint(self, filename):
        try:
            checkpoint = torch.load(filename)
            return checkpoint
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()



    def predict(self, batch_tuple, cuda, unsort=True):
        predictions = OrderedDict()

        for representation, batch in zip(self.models.keys(), batch_tuple):
            anchor_input = None
            total_logits = None

            for model in self.models[representation]:
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

            predictions[representation] = np.argmax(total_logits.data.cpu().numpy(), axis=1).tolist()


        if unsort:
            _, predictions['ud'], ids = [list(t) for t in zip(*sorted(zip(anchor_input.orig_idx, predictions['ud'], anchor_input.id)))]

        return predictions['ud'], ids




