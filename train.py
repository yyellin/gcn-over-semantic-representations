"""
Train a model on enhanced TACRED

  Original Authors: Wenxuan Zhou, Yuhao Zhang
  Enhanced By: Jonathan Yellin
  Status: prototype

"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
import json
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from utils.ucca_embedding import UccaEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--ucca_embedding_dir', default=r'dataset/ucca-embedding')

parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='GCN number of features.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--prune_k', default=-1, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--max_ud_path', type=int, default=-1, help='Filter sentences with longer UD paths from training; set to -1 for no UCCA path based filtering')
parser.add_argument('--max_ucca_path', type=int, default=-1, help='Filter sentences with longer UCCA paths from training; set to -1 for no UCCA path based filtering')

parser.add_argument('--train_without_shuffling', action='store_true', help='Should we not shuffle?')
parser.add_argument('--mask_in_self_loop', action='store_true', help='Mask in self loop?')
parser.add_argument('--fix_subj_obj_mask_bug', action='store_true', help='Fix subject/object mask bug?')

parser.add_argument('--primary_engine', choices=('spacy', 'corenlp'), default='corenlp', help='Which NLP parse to use?')


parser.add_argument('--coref_dim', type=int, default=0, help='COREF embedding dimension (available only when primary_engine=corenlp).')

parser.add_argument('--ud_heads', action='store_true', help='UD heads')
parser.add_argument('--ucca_heads', action='store_true', help='UCCA single heads')
parser.add_argument('--ucca_multi_heads', action='store_true', help='UCCA multi heads')
parser.add_argument('--sequential_heads', action='store_true', help='sequential heads')


parser.add_argument('--ucca_embedding_dim', type=int, default=0, help='UCCA Path to Root Emdedding vector dimension.')
parser.add_argument('--ucca_embedding_file', default='ucca_path_embeddings', help='UCCA Path to Root Embedding vector file')
parser.add_argument('--ucca_embedding_index_file', default='ucca_path_embedding_index', help='UCCA Path to Root Embedding vector file')
parser.add_argument('--ucca_embedding_ignore', action='store_true', help='Do not initialize UCCA embedding with prepared matrix')
parser.add_argument('--ucca_embedding_source', choices=('min_sub_tree', 'all'), default='min_sub_tree', help='use all embeddings or just those that belong to UCCA\'s min subtree')

parser.add_argument('--entity_fix_csv', type=str, help='correct subj or obj entity identification')


parser.add_argument('--binary_classification', type=str, default=None, help='all relation except that provided to be considered as negtive')

args = parser.parse_args()

if not args.binary_classification is None:
    for label in constant.LABEL_TO_ID.keys():
        if label != args.binary_classification:
            constant.LABEL_TO_ID[label] = 0


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# UCCA Embedding?
ucca_embedding = None
if args.ucca_embedding_dim > 0:
    embedding_file = args.ucca_embedding_dir + '/' + args.ucca_embedding_file
    index_file = args.ucca_embedding_dir + '/' +  args.ucca_embedding_index_file
    ucca_embedding =  UccaEmbedding(args.ucca_embedding_dim, index_file, embedding_file)
    opt['ucca_embedding_vocab_size'] = ucca_embedding.embedding_matrix.shape[0]
    assert ucca_embedding.embedding_matrix.shape[1] == args.ucca_embedding_dim

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
with open(opt['data_dir'] + '/train.json') as infile:
    train_input = json.load(infile)
train_batch = DataLoader(train_input, opt['batch_size'], opt, vocab, evaluation=False, apply_filters=True,
                         ucca_embedding=ucca_embedding)
print("{} batches created for train".format(len(train_batch.data)))

with open(opt['data_dir'] + '/dev.json') as infile:
    dev_input = json.load(infile)
dev_batch = DataLoader(dev_input, opt['batch_size'], opt, vocab, evaluation=True, apply_filters=True,
                       ucca_embedding=ucca_embedding)
print("{} batches created for dev".format(len(dev_batch.data)))

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

# model
if not opt['load']:
    trainer = GCNTrainer(opt, emb_matrix=emb_matrix, ucca_embedding_matrix=ucca_embedding.embedding_matrix if ucca_embedding else None)
else:
    # load pretrained model
    model_file = opt['model_file'] 
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = GCNTrainer(model_opt)
    trainer.load(model_file)   

# The id2label[0] = 'no_relation' assignment is necessary for when --binary_classification is active
id2label = dict([(v,k) for k,v in label2id.items()])
id2label[0] = 'no_relation'

dev_score_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss, _ = trainer.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
        train_loss, dev_loss, dev_f1))
    dev_score = dev_f1
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or dev_score > max(dev_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_score*100))
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))

