"""
Run evaluation with saved models.
"""
import random
import argparse
import csv
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

parser.add_argument('--trace_file_for_misses', type=str, help='When provided misses will be outputed to file')

args = parser.parse_args()

if args.trace_file_for_misses != None:
    if not helper.is_path_exists_or_creatable(args.trace_file_for_misses):
        print(f'"{args.trace_file_for_misses}" is an invalid path. Please supply correct "trace_file_for_misses". Exiting.')
        exit(1)


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
if opt['ucca_dim'] > 0:
    embedding_file = opt['ucca_embedding_dir'] + '/' + opt['ucca_embedding_file']
    index_file = opt['ucca_embedding_dir'] + '/' +  opt['ucca_embedding_index_file']
    ucca_embedding =  UccaEmbedding(args.ucca_dim, index_file, embedding_file)


# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True, ucca_embedding=ucca_embedding)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
all_ids = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _, ids = trainer.predict(b)
    predictions += preds
    all_probs += probs
    all_ids += ids

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

if args.trace_file_for_misses != None:
    print(f'Preparing miss information and writing it to "{args.trace_file_for_misses}"')

    with open(args.trace_file_for_misses, 'w', encoding='utf-8', newline='') as trace_file_for_misses:
        csv_writer = csv.writer(trace_file_for_misses)
        csv_writer.writerow( ['id', 'gold', 'predicted'])

        for gold, prediction, id in zip(batch.gold(), predictions, all_ids):
            if gold != prediction:
                csv_writer.writerow( [id, gold, prediction])




print("Evaluation ended.")

