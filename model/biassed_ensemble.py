"""
biased ensemble evaluation
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import torch
import numpy as np

from model.input import Input
from model.gcn import GCNClassifier

from utils import torch_utils, constant

@dataclass
class ModelStuff:
    representation: str
    dirs: List[str]
    files: List[str]
    data: object


class GCNBiassedEnsembleEvaluator(object):

    def __init__(self, model_stuff_list: List[ModelStuff], biassed_prediction):
        self.id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        self.models = OrderedDict()
        self.biassed_prediction = biassed_prediction

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
        logits = OrderedDict()
        model_predictions = OrderedDict()
        for representation, batch in zip(self.models.keys(), batch_tuple):
            anchor_input = None

            for model in self.models[representation]:
                input, labels = Input.unpack_batch(batch, cuda)

                if anchor_input is None:
                    anchor_input = input
                else:
                    assert input.id == anchor_input.id
                    assert input.orig_idx == anchor_input.orig_idx

                model.eval()
                model_output, _ = model(input)

                if representation not in logits:
                    logits[representation] = model_output
                else:
                    logits[representation] += model_output

        overall_logits = None
        for representation in self.models.keys():
            if overall_logits is None:
                overall_logits = logits[representation]
            else:
                overall_logits += logits[representation]
        overall_predictions =  np.argmax(overall_logits.data.cpu().numpy(), axis=1).tolist()

        for representation in self.models.keys():
            model_predictions[representation] = np.argmax(logits[representation].data.cpu().numpy(), axis=1).tolist()

        for index in range(batch_tuple[0].batch_size):
            individial_model_prediction = []
            for representation in self.models.keys():
                individial_model_prediction.append(model_predictions[representation][index])

            overall_predictions[index] = self.biassed_prediction(individial_model_prediction, overall_predictions[index])

        if unsort:
            _, overall_predictions, ids = [list(t) for t in zip(*sorted(zip(anchor_input.orig_idx, overall_predictions, anchor_input.id)))]


        return overall_predictions, ids




