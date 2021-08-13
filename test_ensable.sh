#!/bin/bash

# This script assumes that we have trained both a UD and a UCCA model 20 times, using model IDs 10 through 29,
# and that the resulting models are available.


LOOP=20
UD_DIR=replace-with-directory-of-saved-ud-models
UCCA_DIR=replace-with-directory-of-saved-ucca-models

for ((RUN=0; RUN<${LOOP}; RUN++))
do
        MODEL_IDS=( $(shuf -i 10-29 -n 4) )

        mkdir -p results/${RUN}
        echo "chosen models for run $RUN: ${MODEL_IDS[@]}" > results/${RUN}/README

        ALL_UD_MODELS=()
        ALL_UCCA_MODELS=()

        for MODEL_ID in ${MODEL_IDS[@]}
        do
            UD_MODEL=${UD_DIR}/saved_models/${MODEL_ID}
            UCCA_MODEL=${UCCA_DIR}/saved_models/${MODEL_ID}
            ALL_UD_MODELS[${#ALL_UD_MODELS[@]}]=${UD_MODEL}
            ALL_UCCA_MODELS[${#ALL_UCCA_MODELS[@]}]=${UCCA_MODEL}
        done

        python eval_ensemble.py ${ALL_UD_MODELS[*]} > results/${RUN}/eval.ud.out
        python eval_ensemble.py ${ALL_UCCA_MODELS[*]} > results/${RUN}/eval.ucca.out


        python3 combinations.py ${MODEL_IDS[*]} |

        while read -r COMBINATION_STR
        do
            read -r -a COMBINATION <<< $COMBINATION_STR
            UD_MODELS=()
            UCCA_MODELS=()
            for MODEL_ID in ${COMBINATION[@]}
            do
                UD_MODEL=${UD_DIR}/saved_models/${MODEL_ID}
                UCCA_MODEL=${UCCA_DIR}/saved_models/${MODEL_ID}
                UD_MODELS[${#UD_MODELS[@]}]=${UD_MODEL}
                UCCA_MODELS[${#UCCA_MODELS[@]}]=${UCCA_MODEL}
            done

            UD_MODEL_ARGS=$(echo ${UD_MODELS[*]} | sed -e 's/\s\+/,/g')
            UCCA_MODEL_ARGS=$(echo ${UCCA_MODELS[*]} | sed -e 's/\s\+/,/g')


            COMBINATION_ID=$(echo ${COMBINATION_STR} | sed -e 's/\s\+/-/g')

            python eval_biassed_ensemble.py ${UD_MODEL_ARGS} ${UCCA_MODEL_ARGS} --strategy ucca_and_ud > results/${RUN}/eval.biassed.out${COMBINATION_ID}
            python eval_biassed_ensemble.py ${UD_MODEL_ARGS} ${UCCA_MODEL_ARGS} --strategy nothing > results/${RUN}/eval.non_biassed.out${COMBINATION_ID}


        done


done
