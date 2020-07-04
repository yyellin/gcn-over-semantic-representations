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
parser.add_argument('model_dirs', nargs='+', help='List of model directories')
parser.add_argument('--model_file', type=str, default='best_model.pt', help='Name of the model 1 file.')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--trace_file_for_misses', type=str, help='When provided misses will be outputed to file')

args = parser.parse_args()

if args.trace_file_for_misses != None:
    if not helper.is_path_exists_or_creatable(args.trace_file_for_misses):
        print(f'"{args.trace_file_for_misses}" is an invalid path. Please supply correct "trace_file_for_misses". Exiting.')
        exit(1)

cuda = False
if not args.cpu and torch.cuda.is_available():
    cuda = True

torch.manual_seed(args.seed)
random.seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

anchor_opt = None

model_files = []
all_batches = []

for model_dir in args.model_dirs:
    model_file = model_dir + '/' + args.model_file
    model_files.append(model_file)

    opt = torch_utils.load_config(model_file)

    if anchor_opt is None:
        anchor_opt = opt
    else:
        if opt['data_dir'] != anchor_opt['data_dir']:
            print('models have different data dir. exiting.')
            exit(1)

        if opt['batch_size'] != anchor_opt['batch_size']:
            print('models use different batch size. exiting.')
            exit(1)

    # Vocab
    vocab_file = model_dir + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

    # UCCA Embedding
    ucca_embedding = None
    if opt['ucca_embedding_dim'] > 0:
        embedding_file = opt['ucca_embedding_dir'] + '/' + opt['ucca_embedding_file']
        index_file = opt['ucca_embedding_dir'] + '/' +  opt['ucca_embedding_index_file']
        ucca_embedding =  UccaEmbedding(opt['ucca_embedding_dim'], index_file, embedding_file)

    data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
    batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True, ucca_embedding=ucca_embedding)
    all_batches.append(batch)


evaluator = GCNEnsembleEvaluator(model_files)

label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
predictions = []
all_probs = []
all_ids = []

batch_iter = tqdm(zip(*all_batches))
for i, batch in enumerate(batch_iter):
    preds, probs, ids = evaluator.predict(batch, cuda)
    predictions += preds
    all_probs += probs
    all_ids += ids

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(all_batches[0].gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

if args.trace_file_for_misses != None:
    print(f'Preparing miss information and writing it to "{args.trace_file_for_misses}"')

    with open(args.trace_file_for_misses, 'w', encoding='utf-8', newline='') as trace_file_for_misses:
        csv_writer = csv.writer(trace_file_for_misses)
        csv_writer.writerow( ['id', 'gold', 'predicted'])

        for gold, prediction, id in zip(all_batches[0].gold(), predictions, all_ids):
            if gold != prediction:
                csv_writer.writerow( [id, gold, prediction])




print("Evaluation ended.")

