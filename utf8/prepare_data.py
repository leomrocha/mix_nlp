"""
File that prepares a train, test, dev dataset separation per different max string lengths

The outputs will be
"""
from collections import OrderedDict
import gzip
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import orjson as json
from random import shuffle
from unicodedata import normalize

try:
    from .utils import *
except:
    # hack to solve issue with ipython executing this import
    from utils import *


def get_txt2txtfiles(basepaths=[], filter_txt="txt2txt.json"):
    """
    Gets the list of all the files
    :param basepaths: an iterable of paths to look for files
    :param filter_txt: the text string that must be present in the filename
    :return: a (flattened) list of absolute paths to all the found files
    """
    files_lists = [get_all_files_recurse(bp) for bp in basepaths]
    all_files = [f for fl in files_lists for f in fl if filter_txt in f]
    return all_files


# MAX_LENS = (0, 64, 128, 256, 384, 512, 768, 1024, 2048)
# MAX_LENS = (0, 256, 384, 512, 768, 1024, 2048)
MAX_LENS = (0, 256, 384, 512)


def _group_foo(x, arr=np.array(MAX_LENS)):
    ml = max(len(x["input"]), len(x["target"]))
    am = np.argmax(arr > ml)
    return arr[am]


def separate_by_strlen(fname, max_lens=MAX_LENS):
    """
    Separates a single file into multiple ones with different string lengths
    The function assumes the input is a json file AND the json format contains:
    {'input': "[TEXT]",
     'target': "[TEXT]"
     }
     and the filename contains the task at hand
    :param fname: absolute path for the file to process
    :param max_lens: a list of the max length of each string for each file
    :return: will write a list of files separating by max(input|output) length of the task in place,

    """
    # print("processing {}".format(fname))
    fopen = open
    if fname.endswith(".gz"):
        fopen = gzip.open
    with fopen(fname, 'rb') as f:
        jf = json.loads(f.read())
        groups = {}
        uniquekeys = set([])
        for k, g in itertools.groupby(jf, _group_foo):
            if k in groups:
                groups[k].extend(list(g))
            else:
                groups[k] = list(g)
            uniquekeys.add(k)
        groups = OrderedDict(sorted(groups.items()))
        # now save the files
        for k, v in groups.items():
            if k == 0:
                continue
            oname = fname.replace(".json", "max-{}.json".format(k))
            with fopen(oname, "wb") as of:
                # print("saving {}".format(oname))
                txt = json.dumps(v)
                of.write(txt)
                of.flush()


def _try_process(fname):
    try:
        separate_by_strlen(fname)
    except Exception as e:
        print("Error processing file: {} \nWith error: {}".format(fname, e))


# TODO make this configurable instead
UD_VERSION = "2.5"
BASEPATH = "/home/leo/projects/Datasets/text"
# BASEPATH = "/media/nfs/Datasets/text"
CONLLU_BASEPATH = os.path.join(BASEPATH, 'UniversalDependencies/ud-treebanks-v{}'.format(UD_VERSION))
GLUE_BASEPATH = os.path.join(BASEPATH, "GLUE")
SUPERGLUE_BASEPATH = os.path.join(BASEPATH, "SuperGLUE")
WIKIMATRIX_BASEPATH = "/media/nfs/Datasets/text/WikiMatrix/v1"

PATHS = [CONLLU_BASEPATH, GLUE_BASEPATH, SUPERGLUE_BASEPATH, WIKIMATRIX_BASEPATH]
# PATHS = [CONLLU_BASEPATH, GLUE_BASEPATH, SUPERGLUE_BASEPATH]


def process(paths=PATHS):

    all_files = get_txt2txtfiles(paths)

    with Pool(processes=cpu_count()-1) as pool:
        res = pool.map(_try_process, all_files)


TRAIN_PATH = os.path.join(BASEPATH, 'train_selected')


def prepare_lm_data_wikimatrix(path=TRAIN_PATH):
    """
    Prepares only language model tasks from the wikimatrix extracting the target only and corrupting it
    The LM task will be now to reconstruct the target from the corrupted entry
    """
    all_files = get_all_files_recurse(path)
    fnames = [f for f in all_files if 'WikiMatrix' in f]
    # print("processing {}".format(fname))
    for fname in fnames:
        with gzip.open(fname, 'rb') as f:
            jf = json.loads(f.read())
            lines = []
            for j in jf:
                d = {'input': j['target'],
                     'src_lang': j['tgt_lang']
                     }
                lines.append(d)
            # now save the files
            oname = fname.replace(".json", "-langmodel.json")
            with gzip.open(oname, "wb") as of:
                # print("saving {}".format(oname))
                txt = json.dumps(lines)
                of.write(txt)
                of.flush()


def prepare_select_all(paths=PATHS, out_dir=TRAIN_PATH, max_len=512):
    # filter the files depending on the name
    files = get_txt2txtfiles(paths, 'txt2txtmax-')
    all_files = []
    # looks for all the files tht are max_len,
    for f in files:
        num = int(f.split('-')[-1].replace('.json', '').replace('.gz', ''))
        if num == max_len or (0 < num <= max_len):
            all_files.append(f)

    print("Preparing {} files of max_len {}".format(len(all_files), max_len))
    laf = len(all_files)
    # params = list(zip(all_files, [out_dir]*laf))
    for f in all_files:
        os.system("cp {} {}".format(f, out_dir))
    # with Pool(processes=cpu_count()-1) as pool:
    #     res = pool.starmap(_try_prepare, params)


SEPARATOR = '█'
OUTPUT_FNAME = '/home/leo/projects/Datasets/text/train_selected_monofile/monofile.txt'


def json2lines(paths=[TRAIN_PATH], ofile=OUTPUT_FNAME, separator=SEPARATOR):
    """

    :param paths: paths where to find files
    :param separator: separator to use for the csv/tsv style. by default is a Glyph that is not used
    :return: one file where to save all the lines
    """
    all_files = get_txt2txtfiles(paths, 'txt2txtmax-')
    all_files = [f for f in all_files if 'langmodel' not in f]
    for fname in all_files:
        fopen = open
        if fname.endswith(".gz"):
            fopen = gzip.open
        with fopen(fname, 'rb') as f:
            jf = json.loads(f.read())
            txt = []
            for j in jf:
                t = separator.join([j['src_lang'], j['tgt_lang'], j['input'], j['target']]) + '\n'
                txt.append(normalize('NFKC', t))
            with open(ofile, "a+") as of:
                # print("saving {}".format(oname))
                of.writelines(txt)
                of.flush()


if __name__ == "__main__":
    process()
