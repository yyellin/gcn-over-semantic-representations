"""
Prepare initial vectors for UCCA embeddings
"""
import argparse
from utils.ucca_embedding import UccaEmbedding
from utils import helper

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('data_dir', help='Enhanced TACRED directory.')
    parser.add_argument('--ucca_embedding_dim', type=int, default=80, help='UCCA Path to Root Emdedding vector dimension.')
    parser.add_argument('--ucca_embedding_dir', default=r'./dataset/ucca-embedding', help='Output vocab directory.')
    parser.add_argument('--ucca_embedding_file', default='ucca_path_embeddings', help='UCCA Path to Root Embedding vector file')
    parser.add_argument('--ucca_embedding_index_file', default='ucca_path_embedding_index', help='UCCA Path to Root Embedding vector file')
    parser.add_argument('--ucca_embedding_source', choices=('min_sub_tree', 'all'), default='min_sub_tree', help='use all embeddings or just those that belong to UCCA\'s min subtree')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'

    embedding_file = args.ucca_embedding_dir + '/' + args.ucca_embedding_file
    index_file = args.ucca_embedding_dir + '/' +  args.ucca_embedding_index_file

    helper.ensure_dir(args.ucca_embedding_dir)

    UccaEmbedding.prepare(args.ucca_embedding_dim, [train_file, dev_file, test_file], index_file, embedding_file, args.ucca_embedding_source)

    return UccaEmbedding(args.ucca_embedding_dim, index_file, embedding_file)



if __name__ == '__main__':
    main()


