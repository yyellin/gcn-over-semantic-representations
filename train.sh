#!/bin/bash

show_help() {
cat << EOF
Usage: ${0##*/} [--ud] [--ucca] [--seq] [--emb] model-id

 optional args:
    --ud         use UD based adjacensy matrix
    --ucca       use UCCA based adjacensy matrix
    --seq        use 'SEQ' based adjacensy matrix
    --emb        use UCCA terminal-to-root embeddings

 positional args:
    model-id     Model ID under which to save models

EOF
}

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

# Initialize all the option variables to ensure we are not contaminated by variables from the environment.
UD=0
UCCA=0
SEQ=0
EMB=0
ID=

while :; do
    case $1 in
        --help)
            show_help && exit
            ;;
        --ud)
            UD=1
            ;;
        --ucca)
            UCCA=1
            ;;
        --seq)
            SEQ=1
            ;;
        --emb)
            EMB=1
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac



    shift
done


ID=$1
[ -z $ID ] && show_help && exit

ADJACENCY_OPTIONS=
((UD == 1)) && ADJACENCY_OPTIONS="--ud_heads"
((UCCA == 1)) && ADJACENCY_OPTIONS="$ADJACENCY_OPTIONS --ucca_multi_heads"
((SEQ == 1)) && ADJACENCY_OPTIONS="$ADJACENCY_OPTIONS --sequential_heads"
[ -z "$ADJACENCY_OPTIONS" ] && ADJACENCY_OPTIONS="--ud_heads" && printf "NO adjancy options provided, using UD\n"

EMBEDDING_OPTIONS=
((EMB == 1)) && EMBEDDING_OPTIONS="--ucca_embedding_dim 80"


printf "using options:"
printf "--id $ID $ADJACENCY_OPTIONS $EMBEDDING_OPTIONS --seed 21213 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003\n"
python train.py --id $ID $ADJACENCY_OPTIONS $EMBEDDING_OPTIONS --seed 21213 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003
