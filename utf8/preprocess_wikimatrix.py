from multiprocessing import Pool, cpu_count

import gzip
import os
import orjson as json
from pycountry import languages

try:
    from .utils import *
except:
    # to solve issue with ipython executing this import
    from utils import *


def _get_langs_from_fname(fname):
    # names have the form:
    #   WikiMatrix.es-sq.tsv.gz
    orig, dest = fname.split('-')
    orig = orig[-2:]
    dest = dest[:2]
    return orig, dest


def wikimatrix_txt2txt(fname, threshold=1.04):
    lines = []
    # get language origin and target
    o_lang, d_lang = _get_langs_from_fname(fname)
    d_lang = d_lang.split('-')[0]
    dest_lang = languages.get(alpha_2=d_lang) if len(d_lang) == 2 else languages.get(alpha_3=d_lang)
    with gzip.open(fname, 'rb') as f:
        for l in f.readlines():
            mt_score, src, tgt = l.decode('utf-8').split('\t')
            # filter out all data that hasn't got threshold score. Recommended from Facebook's paper == 1.04
            mt_score = float(mt_score)
            if threshold > mt_score:
                continue
            d = {'input': "Translate to {}: {}".format(dest_lang, src),
                 'target': tgt,
                 'src_lang': o_lang,
                 'tgt_lang': d_lang
                 }
            lines.append(d)
    jsn = json.dumps(lines)
    saveto = fname.replace('.tsv.gz', '-txt2txt.json.gz')
    with gzip.open(saveto, 'wb') as f:
        # print("saving to {}".format(saveto))
        f.write(jsn)
        f.flush()


def _try_process(fname):
    try:
        wikimatrix_txt2txt(fname)
    except Exception as e:
        print("Error processing file: {} \nWith error: {}".format(fname, e))


WIKIMATRIX_BASEPATH = "/media/nfs/Datasets/text/WikiMatrix/v1"


def wikimedia_process(rootdir=WIKIMATRIX_BASEPATH):
    all_files = get_all_files_recurse(rootdir)
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_try_process, all_files)


if __name__ == '__main__':
    wikimedia_process()