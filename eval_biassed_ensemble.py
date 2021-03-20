"""
Run evaluation using biassed ensemble(specifically UD and UCCA) with saved models.
"""

import random
import argparse
import csv
import json
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.biassed_ensemble import ModelStuff, GCNBiassedEnsembleEvaluator
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from utils.ucca_embedding import UccaEmbedding


parser = argparse.ArgumentParser()
parser.add_argument('ud_model_dirs', help='List of UD model directories separated with a comma')
parser.add_argument('ucca_model_dirs', help='List of UCCA model directories  separated with a comma')
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


ud = ModelStuff('ud', args.ud_model_dirs.split(','), [], None)
ucca = ModelStuff('ucca', args.ucca_model_dirs.split(','), [], None)

for model_stuff in [ud, ucca]:


    for model_dir in model_stuff.dirs:

        opt = None
        model_file = model_dir + '/' + args.model_file
        model_stuff.files.append(model_file)

        if model_stuff.data is None:
            opt = torch_utils.load_config(model_file)

            data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
            with open(data_file) as infile:
                data_input = json.load(infile)

            # Vocab
            vocab_file = model_dir + '/vocab.pkl'
            vocab = Vocab(vocab_file, load=True)
            assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

            # UCCA Embedding
            ucca_embedding = None
            if opt['ucca_embedding_dim'] > 0:
                embedding_file = opt['ucca_embedding_dir'] + '/' + opt['ucca_embedding_file']
                index_file = opt['ucca_embedding_dir'] + '/' + opt['ucca_embedding_index_file']
                ucca_embedding = UccaEmbedding(opt['ucca_embedding_dim'], index_file, embedding_file)

            data = DataLoader(data_input, opt['batch_size'], opt, vocab, evaluation=True, ucca_embedding=ucca_embedding)
            model_stuff.data = data

        else:

            other_opt = torch_utils.load_config(model_file)
            if opt['data_dir'] != other_opt['data_dir']:
                print('models have different data dir. exiting.')
                exit(1)

            if opt['batch_size'] != other_opt['batch_size']:
                print('models use different batch size. exiting.')
                exit(1)



evaluator = GCNBiassedEnsembleEvaluator([ud, ucca])

label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
predictions = []
all_ids = []

for i, batch_tuple in enumerate(zip(ud.data, ucca.data)):
    preds, ids = evaluator.predict(batch_tuple, cuda)
    predictions += preds
    all_ids += ids

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(ud.data.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))


if args.trace_file_for_misses != None:
    print(f'Preparing miss information and writing it to "{args.trace_file_for_misses}"')

    with open(args.trace_file_for_misses, 'w', encoding='utf-8', newline='') as trace_file_for_misses:
        csv_writer = csv.writer(trace_file_for_misses)
        csv_writer.writerow( ['id', 'gold', 'predicted'])

        for gold, prediction, id in zip(ud.data.gold(), predictions, all_ids):
            if gold != prediction:
                csv_writer.writerow( [id, gold, prediction])




print("Evaluation ended.")

