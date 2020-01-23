import os
from sortedcontainers import SortedDict
import numpy as np
import pyconll
import pyconll.util
import pickle
import ntpath

UD_VERSION = "2.5"
# list of languages that won't be used:
# this is because I'll be using only the first 2 segments of UTF-8, so the idea is that
# Maybe blacklist
MAYBE_BLACKLIST = []
# ["Kurmanji", "Urdu", "Indonesian", "Coptic-Scriptorium",
# "Kazakh", "Marathi", "Tamil", "Thai", "Warlpiri"]
LANG_TOKENS_BLACKLIST = [
    # "Hindi", "Chinese", "Korean", "Tagalog", "Vietnamese", "Telugu", "Uyghur", "Cantonese", "Japanese", "ar_nyuad-ud",
                         "myv_jr-ud", "br_keb-ud"]  # last 2 -> processing errors with pyconll
BLACKLIST = MAYBE_BLACKLIST + LANG_TOKENS_BLACKLIST


# from https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_all_files_recurse(rootdir):
    allfiles = []
    for root, directories, filenames in os.walk(rootdir):
        for filename in filenames:
            allfiles.append(os.path.join(root, filename))
    return allfiles


def filter_conllu_files(conllufiles, blacklist):
    prefiltered_conllu = []
    for f in conllufiles:
        todel = list(filter(lambda bl: bl in f, blacklist))
        if len(todel) == 0:
            prefiltered_conllu.append(f)
    conllu_train = [f for f in prefiltered_conllu if "-train" in f]
    conllu_test = [f for f in prefiltered_conllu if "-test" in f]
    conllu_dev = [f for f in prefiltered_conllu if "-dev" in f]
    return conllu_train, conllu_test, conllu_dev


def conllu_txt2txt():
    pass