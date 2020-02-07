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


def wikimatrix_txt2txt(fname, threshold=1.04, max_sentence_len=384):
    # TODO reject lines that we can't encode correctly ...
    lines = []
    # get language origin and target
    o_lang, d_lang = _get_langs_from_fname(fname)
    d_lang = d_lang.split('-')[0].strip()
    dest_lang = languages.get(alpha_2=d_lang) if len(d_lang) == 2 else languages.get(alpha_3=d_lang)
    dest_lang = dest_lang.name
    with gzip.open(fname, 'rb') as f:
        for l in f.readlines():
            mt_score, src, tgt = l.decode('utf-8').split('\t')
            # filter out all data that hasn't got threshold score. Recommended from Facebook's paper == 1.04
            mt_score = float(mt_score)
            if threshold > mt_score:
                continue
            if max_sentence_len > 1 and (len(src) > max_sentence_len or len(tgt) > max_sentence_len):
                # avoid processing things that will make the life harder later (time)
                continue
            d = {'input': src,
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
# list of blacklisted languages from the current research, this is due to resources availability only.
MAYBE_BLACKLIST_LANGS = ['ceb', ]
BLACKLIST_LANGS = ['ar', 'as', 'azb', 'bn', 'bp', 'ckb', 'eo', 'ew', 'fa', 'fo', 'gom', 'gu', 'hi', 'hu', 'id', 'ilo',
                   'ja', 'ka', 'kk', 'ko', 'lmo', 'ml', 'mr', 'mwl', 'ne', 'pa', 'py', 'sh', 'si', 'ta', 'te', 'th',
                   'tl', 'ur', 'vi',
                   'wuu', 'yi', 'zb', 'zh'
                   ] + MAYBE_BLACKLIST_LANGS


def wikimatrix_process(rootdir=WIKIMATRIX_BASEPATH):
    all_files = get_all_files_recurse(rootdir)
    # make sure to just filter by the language part of the filename, and not something else by mistake
    blacklist = [b + '-' for b in BLACKLIST_LANGS] + ['-' + b for b in BLACKLIST_LANGS]
    all_files = filter_files(all_files, blacklist)
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_try_process, all_files)


def check_encoding_works(char2int, fname, acceptance_criteria=0.01, nlines=10):
    """
    Checks the number of codepoints that are unknown in the input file and prints the stats and filename.
    Assumes that is a WikiMatrix single file

    :param char2int: the dictionary mapping of the characters to int
    :param fname: the filename to check
    :param acceptance_criteria: the acceptance criteria in ratio of failed codepoints over total codepoints in the file
    WARNING there are files taht contain foreign language (for example fi-sk translation contains japanese) so this can
    make a mess, advice to filter those lines in the line by line editing of the encoding step instead
    :param nlines: check onli the first nlines
    :return: True if file accepted, False if not, and a tuple of candidate languages (if False one of those might have
    an encoding problem)
    """
    # TODO
    unk_count = 0
    char_count = 0.0000001
    chars = set([])

    with gzip.open(fname, 'rb') as f:
        lines = f.readlines()
        lcount = 0
        for l in lines:
            if nlines > 0 and lcount > nlines:
                break
            nlines += 1
            cset = set(list(l.decode('utf-8')))
            chars.update(cset)
            for c in cset:
                if c not in char2int:
                    unk_count += 1
            char_count += len(cset)  # is not a fair sum, but makes a nice check this way to filter some languages

    # for c in chars:
    #     if c not in char2int:
    #         unk_count += 1
    ratio = unk_count / char_count
    accept = True if ratio < acceptance_criteria else False
    print("Accept: {} | chars set: {} | chars count: {} | unk_chars = {} | ratio = {}  | fname = {}".format(accept,
                                                                                                            len(chars),
                                                                                                            char_count,
                                                                                                            unk_count,
                                                                                                            ratio,
                                                                                                            path_leaf(
                                                                                                                fname)))
    return accept, _get_langs_from_fname(fname), fname


if __name__ == '__main__':
    wikimatrix_process()
