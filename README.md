Paths to Relation Extraction through Semantic Structure
==========
This repo contains the code for the paper [Paths to Relation Extraction through Semantic Structure](https://aclanthology.org/2021.findings-acl.231/), which shows that semantic language representations, like UCCA, can be as helpful for the task of Relation Extraction (RE) as their syntactic representation counterparts (specifically, UD v1)

This examination leverages the  work of Jhang, Chi and Manning in their paper [Graph Convolution over Pruned Dependency Trees Improves Relation Extraction](https://nlp.stanford.edu/pubs/zhang2018graph.pdf), and uses an extension of their initial code [GCN over pruned trees](https://github.com/qipeng/gcn-over-pruned-trees). Indeed this repo started out (and remains) a fork of theirs. 

Jhang, Chi and Manning differentiate between a regular Graph Convolution Network (GCN) and a Contextualized Graph Convolution Network (C-GCN) that bbbbbemploys an RNN as an initial deep network block. My work focuses primarily on the C-GCN model, which I often also refer to as GCN, ignoring this distinction.

Please refer to  [Paths to Relation Extraction through Semantic Structure](https://github.com/yyellin/gcn-over-semantic-representations/blob/master/Paths_to_Relation_Extraction_through_Semantic_Structures.pdf) for details about the model.


## Prerequisites
The module have been tested on the following environment:
1. Debian 10 (will work on other flavors of Linux) with at least 10G of RAM
2. Python 3.7.3
4. CUDA version 10.0 and 10.1
5. RTX 2070 and RTX 2080


## Environment Setup
It is strongly recommended to run the modules in this project using a clean python virtual-env. Run the following commands to set this up:

```bash
python3 -m venv /path/to/virtual/env
source /path/to/virtual/env/bin/activate
pip install --upgrade pip
pip install wheel
```
Clone this module from github, and cd into the module directory:
```bash
git clone https://github.com/yyellin/gcn-over-semantic-representations.git
cd gcn-over-semantic-representations
```
Now you are ready to install the module dependencies, by running:
```bash
pip install -r requirements.txt
```

## Data Preparation

**Enhanced TACRED**
The code requires an enhanced version of TACRED. Carefully follow the instructions on my  [TACRED Enrichment repo](https://github.com/yyellin/tacred-enrichment) to prepare this enhanced version. 
Once you have the enhanced TACRED data, place the JSON files under the directory `dataset/tacred`.

**GloVe vectors**
Again from the `gcn-over-semantic-representations` directory,  download and unzip GloVe vectors from the Stanford NLP group website, with:
```
bash get_glove.sh
```

**Vocabulary, initial word vector and initial UCCA embedding vector preparation**
Still from the `gcn-over-semantic-representations` directory,   prepare the  vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```
This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

Now  prepare the UCCA embedding  vectors by running:
```
python prepare_ucca_emb.py  dataset/tacred/ dataset/ucca-embedding
```
This will write the UCCA embedding representation to the `dataset/ucca-embedding`.


## Training & Evaluation

#### Training Options
The internal `train.py` module exposes a large number of parameters to control the training process. I provide a `train.sh` script that limits the number of exposed parameters to the following set:

The following optional arguments specify which matrix adjacency regime to use. It is of course possible to use any combination of regimes.
`--ud`: determines whether the UD based adjacency matrix will be used
`--ucca`: determines whether the UCCA based adjacency matrix will be used
`--seq`: determines whether to use the synthetic adjacency matrix , which represents a bi-lexical tree, where each token is the head of the subsequent token in the sentence.

In addition to the matrix adjacency arguments, an additional argument determines whether UCCA embedding should be used:
`--emb`:  when provided UCCA embedding representations will be used to augment the word embeddings.

#### Run Training

Use the `train.sh` script with arguments of your choice as above, remembering to also include a positional argument for the model ID, as in the following example:
```
bash train.sh --ud --ucca --seq --emb 20
```
In this example invocation, model checkpoints and logs will be saved to `./saved_models/20`.

#### Run Evaluation

To run evaluation on the test set, run:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

#### Results
The  [Paths to Relation Extraction through Semantic Structure](https://github.com/yyellin/gcn-over-semantic-representations/blob/master/Paths_to_Relation_Extraction_through_Semantic_Structures.pdf) paper contains an extensive result breakdown analysis; the results when running evaluation and training using the `train.sh` script (which uses a set seed value of 21213), using different combinations of arguments, are as follows:

| Adjacency Regime | Using UCCA Embeddings? | F1 score |
| ---------------- | ---------------------- | -------- |
| UD + UCCA + SEQ  | Yes                    | 67.50    |
| UCCA             | Yes                    | 66.34    |
| UCCA             | No                     | 65.96    |
| UD               | No                     | 65.39    |



## Citation

```
@inproceedings{yellin-abend-2021-paths,
    title = "Paths to Relation Extraction through Semantic Structure",
    author = "Yellin, Jonathan  and
      Abend, Omri",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.231",
    doi = "10.18653/v1/2021.findings-acl.231",
    pages = "2614--2626",
}
```

## License

All work contained in this package is licensed under the Apache License, Version 2.0. 
