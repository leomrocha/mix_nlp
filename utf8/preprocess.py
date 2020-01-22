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

COLA_BASEPATH = os.path.join(GLUE_BASEPATH, "CoLA")
###
# CoLA

def cola_txt2txt(fpath, saveto):
    """
    CoLA text definition:
    Each line in the .tsv files consists of 4 tab-separated columns.
    Column 1:	the code representing the source of the sentence.
    Column 2:	the acceptability judgment label (0=unacceptable, 1=acceptable).
    Column 3:	the acceptability judgment as originally notated by the author.
    Column 4:	the sentence.
    :param fpath: input path of the text file to process
    :return: a json description of the task, per each line in the input:
    { 'input': ....
      'target': [unacceptable|acceptable]
    }
    """
    lines = []
    with open(fpath) as f:
        # print("opening {}".format(fpath))
        for l in f.readlines():
            e = l.split('\t')
            try:
                tgt = "acceptable" if int(e[1]) == 1 else "unacceptable"
                input = "CoLA acceptability of: {}".format(e[3].replace('\n', ''))
                d = {'input': input, 'target': tgt}
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()
    # return jsn


def cola_process(base_path=COLA_BASEPATH):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.tsv')]
    outfiles = [f.replace('.tsv', '-txt2txt.json') for f in files]
    params = zip(files, outfiles)
    with Pool() as pool:
        res = pool.starmap(cola_txt2txt, params)


###
# MNLI
MNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "MNLI/original")


def mnli_txt2txt(fpath, saveto):
    """
    MNLI text definition:
    Each line in the .jsonl files consists of the following fields:

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
            e = json.loads(l)
            try:
                sentence_1 = e['sentence1']
                sentence_2 = e['sentence2']
                d = {
                    'input': "task: MNLI | Sentence 1: {} | Sentence 2: {}".format(sentence_1, sentence_2),
                    'target': e['gold_label'],
                    'input_sentence_1': "task: MNLI parse tree of: {}".format(sentence_1),
                    'input_sentence_2': "task: MNLI parse tree of: {}".format(sentence_2),
                    'parse_target_1': e['sentence1_parse'],
                    'parse_target_2': e['sentence2_parse'],
                }
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def mnli_process(base_path=MNLI_BASEPATH):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.jsonl')]
    outfiles = [f.replace('.jsonl', '-txt2txt.json') for f in files]
    params = zip(files, outfiles)
    with Pool() as pool:
        res = pool.starmap(mnli_txt2txt, params)



###
# MRPC
MRPC_BASEPATH = os.path.join(GLUE_BASEPATH, "MRPC")


def mrpc_txt2txt(fpath, saveto):
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
        i = 0
        for l in f.readlines():
            if i ==0:
                # ignore first line
                continue
            i+=1
            try:
                label, _, _, sentence_1, sentence_2 = l.split('\t')
                d = { 'input': "Are sentences: {} and: {} equivalent?".format(sentence_1, sentence_2),
                      'target': "Yes, Semantically equivalent" if int(label) == 1 else "No, Not equivalent"
                    }
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def mrpc_process(base_path=MRPC_BASEPATH):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.tsv')]
    outfiles = [f.replace('.tsv', '-txt2txt.json') for f in files]
    params = zip(files, outfiles)
    with Pool() as pool:
        res = pool.starmap(mnli_txt2txt, params)


###
# QNLI
QNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "QNLI")


def qnli_txt2txt(fpath, saveto):
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
                index, question, answer, label = l.split('\t')
                d = {
                      'input': "Are these sentences entailed? Question: {}  Answer: {}".format(question, answer),
                      'target': label.replace("_", " ")
                }
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def qnli_process(base_path=QNLI_BASEPATH):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.tsv')]
    outfiles = [f.replace('.tsv', '-txt2txt.json') for f in files]
    params = zip(files, outfiles)
    with Pool() as pool:
        res = pool.starmap(qnli_txt2txt, params)



###
# QQP
QQP_BASEPATH = os.path.join(GLUE_BASEPATH, "QQP")


def qqp_txt2txt(fpath, saveto):
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
                _, _, _, question_1, question_2, is_duplicate = l.split('\t')
                d = {
                    'input': "Are these questions duplicated? Question: {}  Answer: {}".format(question_1, question_2),
                    'target': "Duplicates" if int(is_duplicate) == 1 else "Not duplicates"
                }
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def qqp_process(base_path=QQP_BASEPATH):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.tsv')]
    outfiles = [f.replace('.tsv', '-txt2txt.json') for f in files]
    params = zip(files, outfiles)
    with Pool() as pool:
        res = pool.starmap(_txt2txt, params)


###
#
_BASEPATH = os.path.join(GLUE_BASEPATH, "")


def _txt2txt(fpath, saveto):
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
                d = {

                }
                lines.append(d)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def _process(base_path=_BASEPATH):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.jsonl')]
    outfiles = [f.replace('.jsonl', '-txt2txt.json') for f in files]
    params = zip(files, outfiles)
    with Pool() as pool:
        res = pool.starmap(_txt2txt, params)
