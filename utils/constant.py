"""
Define constants

  Original Authors: Wenxuan Zhou, Yuhao Zhang
  Enhanced By: Jonathan Yellin
  Status: prototype

"""

EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

ALL_NER_TYPES = set(SUBJ_NER_TO_ID.keys()).union( set(OBJ_NER_TO_ID.keys()) ).union( set(NER_TO_ID.keys()) )

COREF_TO_ID = {
    'M_OBJ' :  {ner_type:index for index, ner_type in enumerate(ALL_NER_TYPES)},
    'M_SUBJ':  {ner_type:index+len(ALL_NER_TYPES) for index, ner_type in enumerate(ALL_NER_TYPES)},
    'NO_M' :   {ner_type:index+len(ALL_NER_TYPES)*2 for index, ner_type in enumerate(ALL_NER_TYPES)},
}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

INFINITY_NUMBER = 1e12

SPACY_POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, '.': 2, ',': 3, '-LRB-': 4, '-RRB-': 5, '``': 6, '""': 7, '\'\'': 8, ':': 9, '$': 10, '#': 11, 'AFX': 12, 'CC': 13, 'CD': 14, 'DT': 15, 'EX': 16, 'FW': 17, 'HYPH': 18, 'IN': 19, 'JJ': 20, 'JJR': 21, 'JJS': 22, 'LS': 23, 'MD': 24, 'NIL': 25, 'NN': 26, 'NNP': 27, 'NNPS': 28, 'NNS': 29, 'PDT': 30, 'POS': 31, 'PRP': 32, 'PRP$': 33, 'RB': 34, 'RBR': 35, 'RBS': 36, 'RP': 37, 'SP': 38, 'SYM': 39, 'TO': 40, 'UH': 41, 'VB': 42, 'VBD': 43, 'VBG': 44, 'VBN': 45, 'VBP': 46, 'VBZ': 47, 'WDT': 48, 'WP': 49, 'WP$': 50, 'WRB': 51, 'ADD': 52, 'NFP': 53, 'GW': 54, 'XX': 55, 'BES': 56, 'HVS': 57, '_SP': 58}

SPACY_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, '': 2, 'PERSON': 3, 'NORP': 4, 'FAC': 5, 'ORG': 6, 'GPE': 7, 'LOC': 8, 'PRODUCT': 9, 'EVENT': 10, 'WORK_OF_ART': 11, 'LAW': 12, 'LANGUAGE': 13, 'DATE': 14, 'TIME': 15, 'PERCENT': 16, 'MONEY': 17, 'QUANTITY': 18, 'ORDINAL': 19, 'CARDINAL': 20}

UCCA_DEP_TO_ID = {'': 0, 'P': 1, 'S': 2, 'A': 3, 'D': 4, 'C': 5, 'E': 6, 'N': 7, 'R': 8, 'H': 9, 'L': 10, 'G': 11, 'U': 12, 'F' : 13, 'T': 14}
