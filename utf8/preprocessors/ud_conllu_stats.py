from multiprocessing import Pool, cpu_count

import math
import os
import sys
import orjson as json
import pyconll
import pyconll.util
from pycountry import languages

try:
    from utf8.utils import *
except:
    # to solve issue with ipython executing this import
    from utils import *

try:
    from .preprocess_conllu import *
    # from preprocessors.preprocess_conllu import *
except:
    from preprocess_conllu import *

import pandas as pd
import numpy as np
from scipy import stats

# UD_VERSION = "2.6"
# BASEPATH = "/home/leo/projects/Datasets/text"
# CONLLU_BASEPATH = os.path.join(BASEPATH, 'UniversalDependencies/ud-treebanks-v{}'.format(UD_VERSION))
CONLLU_BASEPATH = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.6"

#
DISTRIBUTIONS = {"norm": stats.norm,
                 "skewnorm": stats.skewnorm,
                 "gennorm": stats.gennorm,
                 "beta": stats.beta,
                 "betaprime": stats.betaprime,
                 }

rootdir = CONLLU_BASEPATH
blacklist = []  # BLACKLIST
allconll = get_all_files_recurse(rootdir)
train, test, dev = filter_conllu_files(allconll, blacklist)


def conllu_get_fields(fname):
    """
    Processes one conllu file
    :param fname: absolute path to the conllu file
    :return:
    """
    conll = pyconll.load_from_file(fname)
    upos = []
    xpos = []
    deprel = []
    sentences = []
    forms = []

    src_lang = path_leaf(fname).split('_')[0]
    for sen in conll:
        sentences.append((src_lang, sen.text))
        try:
            forms.extend([t.form for t in sen._tokens])
        except:
            pass
        try:
            sen_upos = [t.upos for t in sen._tokens]
            upos.append((src_lang, sen.text, tuple(sen_upos)))
        except:
            pass
        try:
            sen_xpos = [t.xpos for t in sen._tokens]
            xpos.append((src_lang, sen.text, tuple(sen_xpos)))
        except:
            pass
        try:
            sen_deprel = [t.deprel for t in sen._tokens]
            deprel.append((src_lang, sen.text, tuple(sen_deprel)))
        except:
            pass

    return (set(upos), len(upos)), (set(xpos), len(xpos)), (set(deprel), len(deprel)), (
    set(sentences), len(sentences)), (set(forms), len(forms))


def _try_get_2list(fname):
    try:
        return conllu_get_fields(fname)
    except Exception as e:
        print("Error processing file: {} \nWith error: {}".format(fname, e))


def conllu_process_get_2list(rootdir=CONLLU_BASEPATH, blacklist=BLACKLIST):
    allconll = get_all_files_recurse(rootdir)
    train, test, dev = filter_conllu_files(allconll, blacklist)
    all_files = train + test + dev
    #     print(all_files)

    with Pool(processes=cpu_count()) as pool:
        res = pool.map(_try_get_2list, all_files)
        return res


def extract_data_from_fields(results):
    upos_data = []
    deprel_data = []
    sentences_data = []
    forms_data = []

    for r in results:
        upos_val, xpos_val, deprel_val, sentences_val, forms_val = r
        #     print("lala 1")
        forms_data.extend(forms_val[0])
        for val in upos_val[0]:
            #         print(val)
            lang1, txt1, upos = val
            upos_data.append((lang1, txt1, upos, len(upos)))
        for lang3, txt3, deprel in deprel_val[0]:
            deprel_data.append((lang3, txt3, deprel, len(deprel)))
        for lang4, txt4 in sentences_val[0]:
            sentences_data.append((lang4, txt4, len(txt4)))

    return upos_data, deprel_data, sentences_data, forms_data


def get_best_distribution(data, distributions=DISTRIBUTIONS):
    dist_results = []
    params = {}
    for dist_name, dist in distributions.items():
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        #         print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, D, p))

    # select the best fitted distribution
    # store the name of the best fit and its p value
    best_dist, D, best_p = max(dist_results, key=lambda item: item[2])
    #     print("Best fitting distribution: "+str(best_dist))
    #     print("Best p value: "+ str(best_p))
    #     print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def compute_distributions(upos_data, deprel_data, sentences_data, langs=None):

    df_upos = pd.DataFrame(upos_data, columns=["lang", "text", "upos", "upos_len"])
    df_deprel = pd.DataFrame(deprel_data, columns=["lang", "text", "deprel", "deprel_len"])
    df_txt = pd.DataFrame(sentences_data, columns=["lang", "text", "text_len"])

    if langs is None:
        langs = sorted(df_upos['lang'].unique())

    langs_data = {}

    for lang in langs:
        try:
            #         fig, ax = plt.subplots()
            dest_lang = languages.get(alpha_2=lang) if len(lang) == 2 else languages.get(alpha_3=lang)
            dest_lang = dest_lang.name
            lng_upos_len = df_upos.loc[df_upos['lang'] == lang]['upos_len']
            lng_deprel_len = df_deprel.loc[df_deprel['lang'] == lang]['deprel_len']
            lng_text_len = df_txt.loc[df_txt['lang'] == lang]['text_len']

            langs_data[lang] = {
                'lang': dest_lang,
                'upos_len': lng_upos_len,
                'upos_distrib': get_best_distribution(lng_upos_len),
                'deprel_len': lng_deprel_len,
                'deprel_distrib': get_best_distribution(lng_deprel_len),
                'text_len': lng_text_len,
                'text_distrib': get_best_distribution(lng_text_len),
            }

        except Exception as e:
            print("Error processing lang {} with Exception {}".format(lang, e))
            pass
    return langs_data


def compute_stats(langs_data):
    # compute statistics for each language based on the found data distribution
    # estimate CDF and length to capture: 50, 75, 85, 90, 95, 97, 98, 99, 99.5, 99.9% of data -> shall this be
    pass


def main(blacklist=[]):
    res = conllu_process_get_2list(blacklist=blacklist)
    upos_data, deprel_data, sentences_data, forms_data = extract_data_from_fields(res)
    langs_data = compute_distributions(upos_data, deprel_data, sentences_data)
