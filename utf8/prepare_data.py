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
MAX_LENS = (0, 256, 384, 512, 768, 1024, 2048)


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
            oname = fname.replace(".json", "max-{}.json".format(k))
            with fopen(oname, "wb") as of:
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


def process(paths=PATHS):

    all_files = get_txt2txtfiles(paths)

    with Pool(processes=cpu_count()-1) as pool:
        res = pool.map(_try_process, all_files)


if __name__ == "__main__":
    process()
