# general imports for file manipulation
import chardet
from collections import defaultdict, Counter
import os
import sys
import gzip
import zipfile  # to read the zipped gutenberg text files
import rdflib  # to read the rdf directory of the gutenberg files
from pathlib import Path  # to deal with file paths, naming and other things in a platform independent manner
import numpy as np
# NLP imports
import spacy
from spacy.attrs import ORTH, NORM, TAG
# language imports
from pycountry import languages  # to deal with language naming conventions
# multiprocessing imports
from multiprocessing import Pool, cpu_count
try:
    import orjson as json
    # import ujson as json
except:
    import json

import re

# from https://stackoverflow.com/a/1007615/4099701
def urlify(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    return s

BASE_DIR = "/home/leo/projects/AI/Datasets/text/Gutenberg/Gutenberg/stats_results/all_results"
CLEAN_DIR = "/home/leo/projects/AI/Datasets/text/Gutenberg/Gutenberg/stats_results/clean_results"


def clean_file(base_dir, clean_dir, fname):
    # delete duplicate keys
    try:
        oname = fname.replace('_all', '_all_clean')
        with gzip.open(Path(base_dir) / fname, "rb") as fr:
            print(f"cleaning {fname}")
            jsn = json.loads(fr.read())
            jsn['stats_data'].pop('by_token', None)
            lang = jsn["metadata"]["language"][0]
            author = urlify(jsn["metadata"]["author"])
            opath = Path(clean_dir) / lang / author
            opath.mkdir(parents=True, exist_ok=True)
            # check that path exists
            with gzip.open(opath / oname, "wb") as fw:
                print("Saving to {}".format(oname))
                fw.write(json.dumps(jsn))
                fw.flush()
    except Exception as e:
        print(f"ERROR processing {fname} with e: \n {e}")
    return


def main(base_dir=BASE_DIR, clean_dir=CLEAN_DIR, n_proc=7):
    # get file list
    filelist = os.listdir(base_dir)
    print("file list ready, start processing")
    # pre-process file list and parametrization
    params = [(base_dir, clean_dir, fname) for fname in filelist]
    with Pool(processes=n_proc) as pool:
        res = pool.starmap(clean_file, params)


if __name__ == '__main__':
    main()


