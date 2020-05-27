from multiprocessing import Pool, cpu_count

import gzip
import os
try:
    import orjson as json
except:
    import json
# from pycountry import languages

try:
    from utf8.utils import *
    from utf8.blacklist import *
except:
    # to solve issue with ipython executing this import
    from utils import *
    from blacklist import *


def _get_langs_from_fname(fname):
    # names have the form:
    #   WikiMatrix.es-sq.tsv.gz
    orig, dest = path_leaf(fname).replace("WikiMatrix.", "").replace(".tsv.gz", "").split('-')
    orig = orig.split('_')[0].strip()
    dest = dest.split('_')[0].strip()
    return orig, dest


def wikimatrix_txt2txt(fname, threshold=1.04, max_sentence_len=512):
    # TODO reject lines that we can't encode correctly ...
    lines = []
    # get language origin and target
    o_lang, d_lang = _get_langs_from_fname(fname)
    # assert len(o_lang)
    # if 2 > min(len(o_lang), len(d_lang)) or max(len(o_lang), len(d_lang) > 3:
    #     return
    # dest_lang = languages.get(alpha_2=d_lang) if len(d_lang) == 2 else languages.get(alpha_3=d_lang)
    # dest_lang = dest_lang.name
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


def wikimatrix_process(rootdir=WIKIMATRIX_BASEPATH):
    all_files = get_all_files_recurse(rootdir)
    # make sure to just filter by the language part of the filename, and not something else by mistake
    blacklist = [b + '-' for b in BLACKLIST_LANGS] + ['-' + b for b in BLACKLIST_LANGS]
    all_files = [f for f in filter_files(all_files, blacklist) if f.endswith(".tsv.gz")]
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_try_process, all_files)

def extract_charset(fname):
    try:
        charset = set([])
        with gzip.open(fname, 'rb') as f:
            lines = f.readlines()
            for txt in lines:
                txt = txt.decode('utf-8')
            charset.update(set(list(txt)))
        saveto = fname.replace('.tsv.gz', '-charset.txt')
        with open(saveto, 'wb') as f:
            # print("saving to {}".format(saveto))
            otxt = ''.join(list(charset)).encode('utf-8')
            f.write(otxt)
            f.flush()
        return charset
    except Exception as e:
        print("Failed extracting chars from {} with error: \n {}".format(fname, e))


def wikimatrix_charset_process(rootdir=WIKIMATRIX_BASEPATH):
    all_files = get_all_files_recurse(rootdir)
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(extract_charset, all_files)


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
