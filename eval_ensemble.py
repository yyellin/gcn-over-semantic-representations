"""
Run evaluation with saved models.
"""
import random
import argparse
import csv
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.ensemble import GCNEnsembleEvaluator
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from utils.ucca_embedding import UccaEmbedding


parser = argparse.ArgumentParser()
parser.add_argument('model1_dir', type=str, help='Directory of model 1.')
parser.add_argument('model2_dir', type=str, help='Directory of model 2.')
parser.add_argument('--model1', type=str, default='best_model.pt', help='Name of the model 1 file.')
parser.add_argument('--model2', type=str, default='best_model.pt', help='Name of the model 2 file.')
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

model1_file = args.model1_dir + '/' + args.model1
opt1 = torch_utils.load_config(model1_file)

model2_file = args.model2_dir + '/' + args.model2
opt2 = torch_utils.load_config(model2_file)

# We need to make sure that the models are compatible - we check this here:
if opt1['data_dir'] != opt2['data_dir']:
    print('models have different data dir. exiting.')
    exit(1)

if opt1['batch_size'] != opt2['batch_size']:
    print('models use different batch size. exiting.')
    exit(1)

# Vocab
vocab_file1 = args.model1_dir + '/vocab.pkl'
vocab1 = Vocab(vocab_file1, load=True)
assert opt1['vocab_size'] == vocab1.size, "Vocab size must match that in the saved model."

vocab_file2 = args.model2_dir + '/vocab.pkl'
vocab2 = Vocab(vocab_file2, load=True)
assert opt2['vocab_size'] == vocab2.size, "Vocab size must match that in the saved model."


# UCCA Embedding
ucca_embedding1 = None
if opt1['ucca_embedding_dim'] > 0:
    embedding_file = opt1['ucca_embedding_dir'] + '/' + opt1['ucca_embedding_file']
    index_file = opt1['ucca_embedding_dir'] + '/' +  opt1['ucca_embedding_index_file']
    ucca_embedding1 =  UccaEmbedding(opt1['ucca_embedding_dim'], index_file, embedding_file)

ucca_embedding2 = None
if opt2['ucca_embedding_dim'] > 0:
    embedding_file = opt2['ucca_embedding_dir'] + '/' + opt2['ucca_embedding_file']
    index_file = opt2['ucca_embedding_dir'] + '/' +  opt2['ucca_embedding_index_file']
    ucca_embedding2 =  UccaEmbedding(opt2['ucca_embedding_dim'], index_file, embedding_file)

data_file1 = opt1['data_dir'] + '/{}.json'.format(args.dataset)
batch1 = DataLoader(data_file1, opt1['batch_size'], opt1, vocab1, evaluation=True, ucca_embedding=ucca_embedding1)

data_file2 = opt2['data_dir'] + '/{}.json'.format(args.dataset)
batch2 = DataLoader(data_file2, opt2['batch_size'], opt2, vocab2, evaluation=True, ucca_embedding=ucca_embedding2)

batch = zip(batch1, batch2)

evaluator = GCNEnsembleEvaluator(opt1, model1_file, opt2, model2_file)

cuda = torch.cuda.is_available()

label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
predictions = []
all_probs = []
all_ids = []

batch_iter = tqdm(batch)
for i, (b1, b2) in enumerate(batch_iter):
    preds, probs, _, ids = evaluator.predict(b1, b2, cuda)
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

