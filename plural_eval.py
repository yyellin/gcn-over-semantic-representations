"""
Run evaluation with saved models.
"""
import random
import argparse
import csv
import json
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from utils.ucca_embedding import UccaEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--plurality', type=int, default=3, help="Evaluate on dev or test.")
parser.add_argument('--trace_file', type=str, help='When provided predictions and gold will be outputed in kind')

args = parser.parse_args()


torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# UCCA Embedding?
ucca_embedding = None
if opt['ucca_embedding_dim'] > 0:
    embedding_file = opt['ucca_embedding_dir'] + '/' + opt['ucca_embedding_file']
    index_file = opt['ucca_embedding_dir'] + '/' +  opt['ucca_embedding_index_file']
    ucca_embedding =  UccaEmbedding(opt['ucca_embedding_dim'], index_file, embedding_file)


# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))

with open(data_file) as infile:
    data_input = json.load(infile)

loaded = DataLoader(data_input, opt['batch_size'], opt, vocab, evaluation=True, ucca_embedding=ucca_embedding)
print("{} batches created for test".format(len(loaded.data)))


helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

all_predictions = []
for i in range(args.plurality):
    all_predictions.append([])


all_ids = []
for i, batch in enumerate(loaded):
    all_preds, ids = trainer.plural_predict(batch, args.plurality)

    for predictions, preds in zip(all_predictions, all_preds):
        predictions += preds

    all_ids += ids

all_prediction_labels = []

for predictions in all_predictions:
    prediction_labels =  [id2label[p] for p in predictions]
    all_prediction_labels.append(prediction_labels)


p, r, f1 = scorer.plural_score(loaded.gold(), all_prediction_labels, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))


if args.trace_file != None:
    print(f'Creating trace file "{args.trace_file}"')

    with open(args.trace_file, 'w', encoding='utf-8', newline='') as trace_file:
        csv_writer = csv.writer(trace_file)

        title = ['id', 'gold']
        title += ['prediction{0}'.format(i+1) for i in range(len(all_prediction_labels)) ]


        csv_writer.writerow(title)


        for i, (id, gold) in enumerate(zip(all_ids, loaded.gold())):

            row = [id, gold]
            for j in range(len(all_prediction_labels)):
                row.append(all_prediction_labels[j][i])

            csv_writer.writerow( row)


print("Evaluation ended.")
