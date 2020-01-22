import itertools
import numpy as np
import torch
import torchtext
import torchtext.datasets as ttdatasets
import orjson as json
from multiprocessing import Pool, cpu_count
import os
import sys

BASEPATH = "/home/leo/projects/Datasets/text"
########################################################################################################################
# GLUE
GLUE_BASEPATH = os.path.join(BASEPATH, "GLUE")


###
# CoLA
COLA_BASEPATH = os.path.join(GLUE_BASEPATH, "CoLA")


def cola_parser(l):
    e = l.split('\t')
    tgt = "acceptable" if int(e[1]) == 1 else "unacceptable"
    src = "CoLA acceptability of: {}".format(e[3].replace('\n', ''))
    d = {'input': src, 'target': tgt}
    return d


###
# MNLI
MNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "MNLI/original")


def mnli_parser(l):
    e = json.loads(l)
    sentence_1 = e['sentence1']
    sentence_2 = e['sentence2']
    d = {
        'input': "task: MNLI | Sentence 1: {} | Sentence 2: {}".format(sentence_1, sentence_2),
        'target': e['gold_label'],
        'input_1': "task: MNLI parse tree of: {}".format(sentence_1),
        'input_2': "task: MNLI parse tree of: {}".format(sentence_2),
        'target_1': e['sentence1_parse'],
        'target_2': e['sentence2_parse'],
    }
    return d


###
# MRPC
MRPC_BASEPATH = os.path.join(GLUE_BASEPATH, "MRPC")


def mrpc_parser(l):
    label, _, _, sentence_1, sentence_2 = l.split('\t')
    d = {'input': "Are sentences: {} and: {} equivalent?".format(sentence_1, sentence_2),
         'target': "Yes, Semantically equivalent" if int(label) == 1 else "No, Not equivalent"
        }
    return d


###
# QNLI
QNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "QNLI")


def qnli_parser(l):
    index, question, answer, label = l.split('\t')
    d = {
        'input': "Are these sentences entailed? Question: {}  Answer: {}".format(question, answer),
        'target': label.replace("_", " ")
    }
    return d


###
# QQP
QQP_BASEPATH = os.path.join(GLUE_BASEPATH, "QQP")


def qqp_parser(l):
    _, _, _, question_1, question_2, is_duplicate = l.split('\t')
    d = {
        'input': "Are these questions duplicated? Question: {}  Answer: {}".format(question_1, question_2),
        'target': "Duplicates" if int(is_duplicate) == 1 else "Not duplicates"
    }
    return d


###
# RTE
RTE_BASEPATH = os.path.join(GLUE_BASEPATH, "RTE")


def rte_parser(l):
    index, sentence_1, sentence_2, label = l.split('\t')
    d = {
        'input': "Are these sentences entailed? Sentence 1: {}  Sentence 2: {}".format(sentence_1, sentence_2),
        'target': label.replace("_", " ")
    }
    return d


###
# SNLI
SNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "SNLI/original")
snli_parser = mnli_parser


###
#
SST2_BASEPATH = os.path.join(GLUE_BASEPATH, "SST-2")


def sst2_parser(l):

    d = {
        'input': ,
        'target':
    }
    return d

###
#
STSB_BASEPATH = os.path.join(GLUE_BASEPATH, "STS-B")

def stsb_parser(l):
    d = {
        'input': ,
        'target':
    }
    return d

###
#
WNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "WNLI")

def wnli_parser(l):
    d = {
        'input': ,
        'target':
    }
    return d


###
#
def _txt2txt(parser, fpath, saveto):
    """
    :param fpath: input path of the text file to process
    :return: a json description of the task, per each line in the input:
    { 'input': ....
      'target': []
    }
    """
    lines = []
    with open(fpath) as f:
        # print("opening {}".format(fpath))
        for l in f.readlines():
            try:
                d = parser(l)
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def _get_params(function, base_path, extension):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(extension)]
    outfiles = [f.replace(extension, '-txt2txt.json') for f in files]
    params = zip([function]*len(files), files, outfiles)
    return params


def _process(params):
    with Pool(processes=cpu_count()) as pool:
        res = pool.starmap(_txt2txt, params)


def process_glue():
    setup_list = [
        (cola_parser, COLA_BASEPATH, '.tsv'),  # CoLA
        (mnli_parser, MNLI_BASEPATH, '.jsonl'),  # MNLI
        (mrpc_parser, MRPC_BASEPATH, '.tsv'),  # MRPC
        (qnli_parser, QNLI_BASEPATH, '.tsv'),  # QNLI
        (qqp_parser, QQP_BASEPATH, '.tsv'),  # QQP
        (rte_parser, RTE_BASEPATH, '.tsv'),  # RTE
        (snli_parser, SNLI_BASEPATH, '.tsv'),  # SNLI
        (sst2_parser, SST2_BASEPATH, '.tsv'),  # SST-2
        (stsb_parser, STSB_BASEPATH, '.tsv'),  # STS-B
        (wnli_parser, WNLI_BASEPATH, '.tsv'),  # WNLI
        ]

    params = []
    for lst in setup_list:
        params.extend(_get_params(*lst))
    _process(params)
