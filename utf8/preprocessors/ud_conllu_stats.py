from multiprocessing import Pool, cpu_count

import copy
import gzip
import math
import os
import sys
# import orjson as json
import json

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


def _get_stats(distrib, distrib_params, data):
    mskv = [None, None, None, None]
    t_mskv = distrib.stats(*distrib_params)
    for i in range(len(t_mskv)):  # mean, variance, skew, kurtosis -> variable length
        mskv[i] = t_mskv[i]
    ret_stats = {
        'mean': mskv[0],  # mean, variance, skew, kurtosis -> variable length
        'variance': mskv[1],
        'skew': mskv[2],
        'kurtosis': mskv[3],
        'median': distrib.median(*distrib_params),
        'std': distrib.std(*distrib_params),
        'intervals': {'99': distrib.interval(0.99, *distrib_params),
                      '98': distrib.interval(0.98, *distrib_params),
                      '95': distrib.interval(0.95, *distrib_params),
                      '90': distrib.interval(0.90, *distrib_params),
                      '85': distrib.interval(0.85, *distrib_params),
                      '80': distrib.interval(0.8, *distrib_params),
                      }
    }
    ret_foo = {'cdf': distrib.cdf(data, *distrib_params),
               'pdf': distrib.pdf(data, *distrib_params)
               }
    return ret_stats, ret_foo


def _get_lang_stats(lang_data, distributions=DISTRIBUTIONS):
    upos_distrib = distributions[lang_data['upos_distrib'][0]]
    upos_distrib_params = lang_data['upos_distrib'][2]
    #     print('upos', upos_distrib, upos_distrib_params)
    upos_data = lang_data['upos_len']
    upos_stats, upos_functions = _get_stats(upos_distrib, upos_distrib_params, upos_data)
    #
    deprel_distrib = distributions[lang_data['deprel_distrib'][0]]
    deprel_distrib_params = lang_data['deprel_distrib'][2]
    #     print('deprel', deprel_distrib, deprel_distrib_params)
    deprel_data = lang_data['deprel_len']
    deprel_stats, deprel_functions = _get_stats(deprel_distrib, deprel_distrib_params, deprel_data)
    #
    text_distrib = distributions[lang_data['text_distrib'][0]]
    text_distrib_params = lang_data['text_distrib'][2]
    #     print('text', text_distrib, text_distrib_params)
    text_data = lang_data['text_len']
    text_stats, text_functions = _get_stats(text_distrib, text_distrib_params, text_data)

    lang_data['upos_stats'] = upos_stats
    lang_data['deprel_stats'] = deprel_stats
    lang_data['text_stats'] = text_stats

    lang_data['upos_functions'] = upos_functions
    lang_data['deprel_functions'] = deprel_functions
    lang_data['text_functions'] = text_functions

    return lang_data


def flatten(lang, d, sep="_"):
    import collections

    obj = collections.OrderedDict()
    obj['lang_code'] = lang
    lang_name = languages.get(alpha_2=lang) if len(lang) == 2 else languages.get(alpha_3=lang)
    obj['lang_name'] = lang_name.name

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj


def stats_dict2table(all_lang_stats):
    upos_stats = []
    deprel_stats = []
    text_stats = []
    for lang, lang_data in all_lang_stats.items():
        upos_row, deprel_row, text_row = stats_dict2rows(lang, lang_data)
        upos_stats.append(upos_row)
        deprel_stats.append(deprel_row)
        text_stats.append(text_row)

    upos_df = pd.DataFrame(upos_stats)
    deprel_df = pd.DataFrame(deprel_stats)
    text_df = pd.DataFrame(text_stats)

    return upos_df, deprel_df, text_df


def stats_dict2rows(lang, lang_data):
    upos_data = flatten(lang, lang_data['upos_stats'])
    deprel_data = flatten(lang, lang_data['deprel_stats'])
    text_data = flatten(lang, lang_data['text_stats'])
    return upos_data, deprel_data, text_data


# convert to json  TODO this must be improved

# This solution is modified from:
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# https://github.com/mpld3/mpld3/issues/434

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (tuple, set)):
            return list(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            obj = obj.to_list()
        return json.JSONEncoder.default(self, obj)


def _recursive_jsonify(dict_data):
    new_dict = {}
    for k, v in dict_data.items():
        k = str(k)  # always convert,
        if isinstance(v, (tuple, set)):
            ov = []
            for t in v:
                if isinstance(t, str):
                    ov.append(t)
                elif isinstance(t, tuple):
                    ov.append([float(i) for i in t])
                else:
                    ov.append(float(t))
            v = ov
        if isinstance(v, pd.Series):
            v = v.to_list()
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if np.issubdtype(type(v), np.number):
            v = float(v)
        if isinstance(v, dict):
            new_dict[k] = _recursive_jsonify(v)
        else:
            new_dict[k] = v
    return new_dict
###


def generate_files(blacklist=[], saveto='conllu_stats.json.zip'):
    res = conllu_process_get_2list(blacklist=blacklist)
    upos_data, deprel_data, sentences_data, forms_data = extract_data_from_fields(res)
    langs_data = compute_distributions(upos_data, deprel_data, sentences_data)

    all_stats = {}

    for lang, lang_data in langs_data.items():
        print('processing {}'.format(lang))
        all_stats[lang] = _get_lang_stats(lang_data)

    all_stats_copy = copy.deepcopy(all_stats)
    all_stats_copy = _recursive_jsonify(all_stats_copy)
    jsn = json.dumps(all_stats_copy)
    # jsn = json.dumps(all_stats_copy, cls=NumpyEncoder)
    # with open('conllu_stats.json', 'w') as f:
    #     f.write(jsn)
    #     f.flush()

    with gzip.open(saveto, 'wb') as f:
        print("Saving to {}".format(saveto))
        f.write(jsn.encode('utf-8'))
        f.flush()

    return all_stats


