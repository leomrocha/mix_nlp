from multiprocessing import Pool, cpu_count

import os
import orjson as json
import pyconll
import pyconll.util

try:
    from .utils import *
except:
    # to solve issue with ipython executing this import
    from utils import *

UD_VERSION = "2.5"
# Maybe blacklist
MAYBE_BLACKLIST = []

LANG_TOKENS_BLACKLIST = [
    "myv_jr-ud", "br_keb-ud"]  # last 2 -> processing errors with pyconll
BLACKLIST = MAYBE_BLACKLIST + LANG_TOKENS_BLACKLIST


def filter_conllu_files(conllufiles, blacklist, extension='.conllu'):
    prefiltered_conllu = []
    for f in conllufiles:
        todel = list(filter(lambda bl: bl in f, blacklist))
        if len(todel) == 0:
            prefiltered_conllu.append(f)
    prefiltered_conllu = [f for f in prefiltered_conllu if f.endswith(extension)]
    conllu_train = [f for f in prefiltered_conllu if "-train" in f]
    conllu_test = [f for f in prefiltered_conllu if "-test" in f]
    conllu_dev = [f for f in prefiltered_conllu if "-dev" in f]
    return conllu_train, conllu_test, conllu_dev


def conllu_txt2txt(fname):
    """
    Processes one conllu file
    :param fname: absolute path to the conllu file
    :return: writes 6 new text-to-text tasks json files from the original conllu one with the same root name
    with contents:
    {'input': "Task: POS Tagging TASK of: {}".format(sentence.text),
     'target': TASK DETAILS
     }
    """
    conll = pyconll.load_from_file(fname)
    # conll_txt = []
    conll_lemma = []
    conll_upos = []
    conll_xpos = []
    conll_feats = []
    conll_head = []
    conll_deprel = []
    # conll_deps = []
    # conll_misc = []
    for sen in conll:
        try:
            sen_lemma = ' '.join([t.lemma for t in sen._tokens])
            conll_lemma.append({'input': "Task: POS Tagging Lemma of: {}".format(sen.text), 'target': sen_lemma})
        except:
            pass
        try:
            sen_upos = ' '.join([t.upos for t in sen._tokens])
            conll_upos.append({'input': "Task: POS Tagging UPOS of: {}".format(sen.text), 'target': sen_upos})
        except:
            pass
        try:
            sen_xpos = ' '.join([t.xpos for t in sen._tokens])
            conll_xpos.append({'input': "Task: POS Tagging XPOS of: {}".format(sen.text), 'target': sen_xpos})
        except:
            pass
        try:
            sen_head = ' '.join([t.head for t in sen._tokens])
            conll_feats.append({'input': "Task: POS Tagging FEATS of: {}".format(sen.text), 'target': feats})
        except:
            pass
        try:
            sen_deprel = ' '.join([t.deprel for t in sen._tokens])
            conll_head.append({'input': "Task: POS Tagging HEAD of: {}".format(sen.text), 'target': sen_head})
        except:
            pass
        try:
            sen_form = [t.form for t in sen._tokens]
            sen_feats = [t.feats for t in sen._tokens]
            feats = "|".join(e[0] + "=" + str(e[1]) for e in list(zip(sen_form, sen_feats))) \
                .replace("{", "").replace("}", "")
            conll_deprel.append({'input': "Task: POS Tagging DEPREL of: {}".format(sen.text), 'target': sen_deprel})
        except:
            pass
        # sen_deps = [t.deps for t in sen._tokens]
        # sen_misc = [t.mis for t in sen._tokens]
    # now save all the files
    data = [
        (conll_lemma, "-PoS-text2text-lemma.json"),
        (conll_upos, "-PoS-text2text-upos.json"),
        (conll_xpos, "-PoS-text2text-xpos.json"),
        (conll_feats, "-PoS-text2text-feats.json"),
        (conll_head, "-PoS-text2text-head.json"),
        (conll_deprel, "-PoS-text2text-deprel.json"),
    ]
    data = [d for d in data if len(d[0]) > 0]

    for lines, name in data:
        jsn = json.dumps(lines)
        saveto = fname.replace(".conllu", name)
        with open(saveto, 'wb') as f:
            # print("saving to {}".format(saveto))
            f.write(jsn)
            f.flush()


def _try_process(fname):
    try:
        conllu_txt2txt(fname)
    except Exception as e:
        print("Error processing file: {} \nWith error: {}".format(fname, e))


BASEPATH = "/home/leo/projects/Datasets/text"
CONLLU_BASEPATH = os.path.join(BASEPATH, 'UniversalDependencies/ud-treebanks-v{}'.format(UD_VERSION))


def conllu_process(rootdir=CONLLU_BASEPATH, blacklist=BLACKLIST):
    allconll = get_all_files_recurse(rootdir)
    train, test, dev = filter_conllu_files(allconll, blacklist)
    all_files = train + test + dev
    # print(all_files)

    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_try_process, all_files)