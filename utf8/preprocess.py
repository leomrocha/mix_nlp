import orjson as json
from multiprocessing import Pool, cpu_count
import os
import ntpath

try:
    from .utils import *
except:
    # hack to solve issue with ipython executing this import
    from utils import *

BASEPATH = "/home/leo/projects/Datasets/text"


########################################################################################################################
###
def _txt2txt(parser, fpath, saveto, ignore_header=True):
    """
    :param fpath: input path of the text file to process
    :return: a json description of the task, per each line in the input:
    { 'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': '',  'input': ....
      'target': []
    }
    """
    lines = []
    with open(fpath) as f:
        print("opening {}".format(fpath))
        i = 0
        for l in f.readlines():
            i += 1
            if ignore_header and i <= 1:
                continue
            try:
                d = parser(l)
                if type(d) is dict:
                    d = [d]  # fixes the issue if there are multiple outputs from the parser, each should be a json line
                for di in d:
                    lines.append(di)
            except Exception as e:
                # TODO handle error here
                pass
    jsn = json.dumps(lines)
    with open(saveto, 'wb') as f:
        print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def _get_params(function, base_path, extension, ignore_header=True):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(extension)]
    outfiles = [f.replace(extension, '-txt2txt.json') for f in files]
    params = zip([function] * len(files), files, outfiles, [ignore_header] * len(files))
    return params


def _process(params):
    with Pool(processes=cpu_count()) as pool:
        res = pool.starmap(_txt2txt, params)


########################################################################################################################
# GLUE
GLUE_BASEPATH = os.path.join(BASEPATH, "GLUE")

###
COLA_BASEPATH = os.path.join(GLUE_BASEPATH, "CoLA")


def cola_parser(l):
    e = l.split('\t')
    src = e[3].replace('\t', '')
    tgt = "acceptable" if int(e[1]) == 1 else "unacceptable"
    d = {'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en',
         'task': ' CoLA Acceptability',
         'input': src,
         'target': tgt
         }
    return d


###
MNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "MNLI/original")


def mnli_parser(l):
    e = json.loads(l)
    sentence_1 = e['sentence1']
    sentence_2 = e['sentence2']
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'MNLI Language Inference',
        'input': "{}\n{}".format(sentence_1, sentence_2),
        'target': e['gold_label'],
    }
    d1 = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'MNLI Parse Tree',
        'input': sentence_1,
        'target': e['sentence1_parse'],
    }
    d1 = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'MNLI Parse Tree',
        'input': sentence_2,
        'target': e['sentence2_parse'],
    }
    return d, d1, d2


###
MRPC_BASEPATH = os.path.join(GLUE_BASEPATH, "MRPC")


def mrpc_parser(l):
    label, _, _, sentence_1, sentence_2 = l.split('\t')
    d = {'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'MRPC Equivallency',
         'input': "{}\n{}".format(sentence_1, sentence_2),
         'target': "Yes, Semantically equivalent" if int(label) == 1 else "No, Not equivalent"
         }
    return d


###
QNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "QNLI")


def qnli_parser(l):
    index, question, answer, label = l.split('\t')
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'QNLI Entailment',
        'input': "{}\n{}".format(question, answer),
        'target': label.replace("_", " ")
    }
    return d


###
QQP_BASEPATH = os.path.join(GLUE_BASEPATH, "QQP")


def qqp_parser(l):
    _, _, _, question_1, question_2, is_duplicate = l.split('\t')
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'QQP Duplication Detection',
        'input': "{}\n{}".format(question_1, question_2),
        'target': "Duplicates" if int(is_duplicate) == 1 else "Not duplicates"
    }
    return d


###
RTE_BASEPATH = os.path.join(GLUE_BASEPATH, "RTE")


def rte_parser(l):
    _, sentence_1, sentence_2, label = l.split('\t')
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'RTE Entailment',
        'input': "{}\n{}".format(sentence_1, sentence_2),
        'target': label.replace("_", " ").capitalize()
    }
    return d


###
SNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "SNLI/original")
snli_parser = mnli_parser

###
SST2_BASEPATH = os.path.join(GLUE_BASEPATH, "SST-2")


def sst2_parser(l):
    txt, tgt = l.split('\t')
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'SNLI Sentiment Analysis',
        'input': txt,
        'target': "Positive" if int(tgt) == 1 else "Negative"
    }
    return d


###
STSB_BASEPATH = os.path.join(GLUE_BASEPATH, "STS-B")


def stsb_parser(l):
    _, _, _, _, _, _, _, sentence1, sentence2, score = l.split('\t')

    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'STS-B Similarity',
        'input': "{}\n{}".format(sentence1, sentence2),
        'target': str(score)
    }
    return d


###
WNLI_BASEPATH = os.path.join(GLUE_BASEPATH, "WNLI")


def wnli_parser(l):
    _, sentence_1, sentence_2, label = l.split('\t')
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'WNLI Entailment',
        'input': "{}\n{}".format(sentence_1, sentence_2),
        'target': "Entailed" if int(label) == 1 else "Not entailed"
    }
    return d


#########################################################
# SuperGLUE
SUPERGLUE_BASEPATH = os.path.join(BASEPATH, "SuperGLUE")
##
BOOLQ_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "BoolQ")


def boolq_parser(l):
    rec = json.loads(l)
    question = rec['question'] + '?'  # set again the capital and the question mark
    question = question.capitalize().replace('??', '?')
    passage = rec['passage']
    answer = "Yes" if rec['label'] else "No"
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'BoolQ Question Answer',
        'input': "{}\n{}".format(passage, question),
        'target': answer
    }
    return d


##
CB_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "CB")


def cb_parser(l):
    e = json.loads(l)
    premise = e['premise']
    hypothesis = e['hypothesis']
    label = e['label']
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'CB Relationship',
        'input': '{} \n {}'.format(premise, hypothesis),
        'target': label.capitalize()
    }
    return d


##
COPA_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "COPA")


def copa_parser(l):
    e = json.loads(l)
    premise = e["premise"]
    choice1 = e["choice1"]
    choice2 = e["choice2"]
    question = e["question"]
    label = e["label"]

    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'COPA Multiple Choice',
        'input': "{}\n{}\n{}\n What is the {}? ".format(question, premise, choice1, choice2),
        'target': "Choice {}".format(int(label) + 1)
    }
    return d


##
# TODO later, this is a tough one (and text will be long)
MULTIRC_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "MultiRC")
# def multirc_parser(l):
#     e = json.loads(l)
#
#     d = {
#         'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'Multiple Choice',
#         'input': ,
#         'target':
#     }
#     return d

##
RECORD_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "ReCoRD")


def record_parser(l):
    e = json.loads(l)
    passage = e['passage']['text']
    query = e['qas'][0]['query']
    answer = e['qas'][0]['answers']['text']
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'ReCoRD Fill Placeholder',
        'input': "{}\n{} ".format(passage, query),
        'target': answer
    }
    return d


##
SG_RTE_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "RTE")


def sg_rte_parser(l):
    e = json.loads(l)
    premise = e['premise']
    hypothesis = e['hypothesis']
    label = e['label']
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'SuperGLUE RTE Entailment',
        'input': "{}\n{}".format(premise, hypothesis),
        'target': label.replace("_", " ").capitalize()
    }
    return d


##
WIC_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "WiC")


def wic_parser(l):
    e = json.loads(l)
    word = e['word']
    sentence1 = e['sentence1']
    sentence2 = e['sentence2']
    label = e['label']
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'WiC Equivallency',
        'input': "Is the word {} in both sentences equivalent? \n {} \n {}".format(word, sentence1, sentence2),
        'target': "Yes" if label else "No"
    }
    return d


##
WSC_BASEPATH = os.path.join(SUPERGLUE_BASEPATH, "WSC")


def wsc_parser(l):
    e = json.loads(l)
    text = e['text']
    tgt1 = e['target']['span1_text']
    tgt2 = e['target']['span2_text']
    label = e['label']
    d = {
        'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': 'WSC Antecedent',
        'input': "In the sentence: {} \n Does {} refer to {}?".format(text, tgt2, tgt1),
        'target': "Yes" if label else "No"
    }
    return d


#########################################################

def rename_files():
    """
    Original files have a simple name that collides when mixing datasets
    """
    setup_list = [
        (COLA_BASEPATH, 'cola'),
        (MNLI_BASEPATH, 'mnli'),
        (MRPC_BASEPATH, 'mrpc'),
        (QNLI_BASEPATH, 'qnli'),
        (QQP_BASEPATH, 'qqp'),
        (RTE_BASEPATH, 'rte'),
        (SNLI_BASEPATH, 'snli'),
        (SST2_BASEPATH, 'sst2'),
        (STSB_BASEPATH, 'stsb'),
        (WNLI_BASEPATH, 'wnli'),
        # SuperGLUE
        (BOOLQ_BASEPATH, 'sg_boolq'),
        (CB_BASEPATH, 'sg_cb'),
        (COPA_BASEPATH, 'sg_copa'),
        (MULTIRC_BASEPATH, 'sg_multirc'),
        (RECORD_BASEPATH, 'sg_record'),
        (SG_RTE_BASEPATH, 'sg_rte'),
        (WIC_BASEPATH, 'sg_wic'),
        (WSC_BASEPATH, 'sg_wsc'),
    ]

    for base_path, prefix in setup_list:
        all_files = get_all_files_recurse(base_path)
        for fname in all_files:
            head, tail = ntpath.split(fname)
            oname = os.path.join(head, prefix + '_' + tail)
            print(oname)
            # os.system("mv {} {}".format(fname, oname))


def process_glue():
    setup_list = [
        (cola_parser, COLA_BASEPATH, '.tsv', True),  # CoLA
        (mnli_parser, MNLI_BASEPATH, '.jsonl', False),  # MNLI
        (mrpc_parser, MRPC_BASEPATH, '.tsv', True),  # MRPC
        (qnli_parser, QNLI_BASEPATH, '.tsv', True),  # QNLI
        (qqp_parser, QQP_BASEPATH, '.tsv', True),  # QQP
        (rte_parser, RTE_BASEPATH, '.tsv', True),  # RTE
        (snli_parser, SNLI_BASEPATH, '.jsonl', True),  # SNLI
        (sst2_parser, SST2_BASEPATH, '.tsv', True),  # SST-2
        (stsb_parser, STSB_BASEPATH, '.tsv', True),  # STS-B
        (wnli_parser, WNLI_BASEPATH, '.tsv', True),  # WNLI
    ]
    params = []
    for lst in setup_list:
        params.extend(_get_params(*lst))
    _process(params)


def process_superglue():
    setup_list = [
        (boolq_parser, BOOLQ_BASEPATH, '.jsonl', False),  # BoolQ
        (cb_parser, CB_BASEPATH, '.jsonl', False),  #
        (copa_parser, COPA_BASEPATH, '.jsonl', False),  #
        # (multirc_parser, MULTIRC_BASEPATH, '.jsonl', False),  #
        (record_parser, RECORD_BASEPATH, '.jsonl', False),  #
        (sg_rte_parser, SG_RTE_BASEPATH, '.jsonl', False),  #
        (wic_parser, WIC_BASEPATH, '.jsonl', False),  #
        (wsc_parser, WSC_BASEPATH, '.jsonl', False),  #
    ]
    params = []
    for lst in setup_list:
        params.extend(_get_params(*lst))
    _process(params)

#########################################################

# SWAGAF_BASEPATH = os.path.join(BASEPATH, "swagaf/data/")

# def swagaf_parser(l):
#     # _,video-id,fold-ind,startphrase,sent1,sent2,gold-source,ending0,ending1,ending2,ending3,label
#     _,video_id,fold_ind,startphrase,sent1,sent2,gold_source,ending0,ending1,ending2,ending3,label = l.split('\t')
#
#     d = {
#         'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': '', 
#         'input': ,
#         'target':
#     }
#     return d
#
# def swagaf_full_parser(l):
#     #video-id,fold-ind,startphrase,gold-ending,distractor-0,distractor-1,distractor-2,distractor-3,gold-source,gold-type,distractor-0-type,distractor-1-type,distractor-2-type,distractor-3-type,sent1,sent2
#      = l.split('\t')
#
#     d = {
#         'src_lang': 'en', 'tgt_lang': 'en', 'task_lang': 'en', 'task': '', 
#         'input': ,
#         'target':
#     }
#     return d
