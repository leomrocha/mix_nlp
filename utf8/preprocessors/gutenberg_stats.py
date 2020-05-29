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
except:
    import json

# local tools imports
# gutenberg project metadata parsing
try:
    from .gutenberg_metainfo import *  # slightly modified script so it works for what I need
except:
    from gutenberg_metainfo import *  # slightly modified script so it works for what I need

# gutenberg specific imports
import gutenberg_cleaner  # to clean headers and footers from gutenberg files, they are NOISE

BASE_DIR = "/media/nfs/Datasets/text/Gutenberg"
# BASE_DIR = "/media/nfs/Datasets/text/Gutenberg/aleph.gutenberg.org"
RDF_TAR_FILE = "/media/nfs/Datasets/text/Gutenberg/rdf-files.tar.bz2"
ZIP_FILE_LIST = "/media/nfs/Datasets/text/Gutenberg/zip_list.txt"
# BASE_RDF_DIR = "/media/nfs/Datasets/text/Gutenberg/rdf_db/cache/epub"


spacy_models = {
    'german': 'de_core_news_md',  # German
    'greek': 'el_core_news_sm',  # Greek
    'english': 'en_core_web_md',  # English
    'spanish': 'es_core_news_md',  # Spanish
    'french': 'fr_core_news_md',  # French
    'italian': 'it_core_news_sm',  # Italian
    'lithuanian': 'lt_core_news_sm',  # Lithuanian
    'norwegian': 'nb_core_news_sm',  # Norwegian Bokmål
    'dutch': 'nl_core_news_sm',  # Dutch
    'portuguese': 'pt_core_news_sm',  # Portuguese
    'multi-lang': 'xx_ent_wiki_sm',  # Multi-Lang
}

spacy_models_2 = {
    'de': 'de_core_news_md',  # German
    'el': 'el_core_news_sm',  # Greek
    'en': 'en_core_web_md',  # English
    'es': 'es_core_news_md',  # Spanish
    'fr': 'fr_core_news_md',  # French
    'it': 'it_core_news_sm',  # Italian
    'lt': 'lt_core_news_sm',  # Lithuanian
    'nb': 'nb_core_news_sm',  # Norwegian Bokmål
    'nl': 'nl_core_news_sm',  # Dutch
    'pt': 'pt_core_news_sm',  # Portuguese
    'xx': 'xx_ent_wiki_sm',  # Multi-Lang
}


def get_file_id(fname):
    """Returns the Gutenberg File ID"""
    pth = Path(fname)
    # as per file structure the filename has some variations but the parent folder is always the ID
    return pth.parent.name


def get_metadata(rdf_tarfile=RDF_TAR_FILE):
    return readmetadata(rdf_tarfile)


def get_nlp_resource(metainfo):
    lang = 'xx'
    try:
        lng = metainfo['language']
        if isinstance(lng, list) or isinstance(lng, tuple) and len(lng > 0):
            lng = lng[0]
        elif isinstance(lng, str):
            pass  # nothing to do here, move along
        else:
            pass  # FUUUUUUU something wrong, but the default value will be multilang anyways
    except:
        # just to avoid issues if there is no language tag, in that case go back to default
        pass
    # loading with shortcut ... maybe will need to use the spacy models dict that I've created earlier, we'll see
    nlp = spacy.load(spacy_models_2[lang])
    return nlp


# count total length of the text
# Separate paragraphs I'll do it by at least a \n\n sequence (might not work for every language .. (I do care but I cant, so somebody that knows the language should correct those)
# count the number of paragraphs ... ?

# separate sentences (again, language depending, I'll do it by . characters and with spacy models ...
# count the number of sentences per paragraph
# separate words -> maybe here would need lemmatization to see some things correctly, but will have issues in non supported languages.
# count number of words tokens per sentence -> tokens is already available in spacy, while words is a bit more .. difficult to define and be sure it works in a coherent way.
# sum number of words tokens per paragraph
# separate chars
# count chars per word, sentence and paragraph
# aggregate all words and count the number of occurrences
# aggregate all the characters and count the number of occurrences
# aggregate and sort results then save zip file with it


# this implementation is quite inefficient as there are a few things done once and again behind the scenes,
# for the moment I don't care as I'm trying to make it work and will see later.
# Nevertheless seems that most of the time is in IO operations

def _get_stats(arr):
    arr = np.array(arr)
    stats = {'total_count': np.sum(arr),
             'min': np.min(arr),
             'max': np.max(arr),
             'mean': np.mean(arr),
             'median': np.median(arr),
             'std': np.std(arr)
             }
    return stats


# this function already works but there is a lot to cleanup and improve
# TODO modify all this to set Custom Attribute Extensions to the document instead:
# https://spacy.io/usage/processing-pipelines#custom-components-attributes
def process_gutenberg_file(fname, metainfo):
    # spacy
    nlp = get_nlp_resource(metainfo)

    nlp.max_length = 1e7  # support larger volumes of text
    with nlp.disable_pipes("ner"):
        # load and clean the file, assume zip as compressing format
        pth = Path(fname)  #
        ftxt = pth.name.replace(".zip", ".txt")  # inside gutenberg zip there should be a .txt file with the same name
        f = zipfile.ZipFile(fname)
        btxt = f.read(ftxt)
        # TODO extract format from metadata (if exists)
        # encoding = 'utf-8'  # assume all is compatible with utf-8, for the moment haven't found one that is not
        # guess the encoding
        enc = chardet.detect(btxt)['encoding']
        txt = btxt.decode(enc)
        txt = gutenberg_cleaner.simple_cleaner(txt)
        # Start analysis
        doc = nlp(txt)  # SpaCy tokenization -> should take out all the steps that are not neede as NER

        ocnt = doc.count_by(ORTH)
        words = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

        tokens = token_count = {doc.vocab.strings[k]: v for k, v in
                                reversed(sorted(ocnt.items(), key=lambda item: item[1]))}
        token_lens = np.array([len(k) for k in token_count.keys()])

        token_stats = _get_stats(token_lens)

        sen_charcount = []  # sentence length in characters
        sen_tok_count = []  # sentence length in tokens
        sen_word_count = []  # sentence length in tokens

        try:
            for s in doc.sents:
                # clen = len(s.string) -> use s.text instead
                clen = len(s.text)
                sen_charcount.append(clen)
                tlen = len(s)
                sen_tok_count.append(tlen)
                ws = [token.text for token in s if not token.is_stop and not token.is_punct and not token.is_space]
                sen_word_count.append(len(ws))

            sen_char_stats = _get_stats(sen_charcount)
            sen_token_stats = _get_stats(sen_tok_count)
            sen_word_stats = _get_stats(sen_word_count)
        except:
            sen_char_stats = {}
            sen_token_stats = {}
            sen_word_stats = {}
            pass  # nothing to see here, move along, this language is not supported

        # slows a lot the processing, but I don't care for the moment
        # I'm doubting using this as much of it is already done in the previous part
        # also makes everything slooower ... need to do it faster so I wont count this stat (even though I do want it)
        # i might be able to do some estimation?, nevertheless
        # count number of paragraphs .. might not work on many languages
        # Somehow this can be done instead as an extension to spacy, to make it faster and process things less times
        paragraphs = [l.strip() for l in txt.split('\n\n') if len(l.strip()) > 0]

        para_char_lens = []
        para_tok_lens = []
        para_word_lens = []
        para_sen_lens = []
        for p in paragraphs:
            d = nlp(p)  # yes, this is slow, something must be done to accelerate it
            para_char_lens.append(len(p))
            para_tok_lens.append(len(list(d)))
            ws = [token.text for token in d if not token.is_stop and not token.is_punct and not token.is_space]
            para_word_lens.append(len(ws))
            para_sen_lens.append(len(list(d.sents)))
        #
        main_para_char_stats = _get_stats(para_char_lens)
        main_para_tok_stats = _get_stats(para_tok_lens)
        main_para_word_stats = _get_stats(para_word_lens)
        main_para_sen_stats = _get_stats(para_sen_lens)

        para_stats = {'char_stats': main_para_char_stats,
                      'token_stats': main_para_tok_stats,
                      'word_stats': main_para_word_stats,
                      'sentence_stats': main_para_sen_stats}

        stats_data = {
            # general data to be able to compute aggregated statistics on the global Gutenberg project files
            'doc': {'paragraph_count': len(paragraphs),
                    'sentence_count': len(list(doc.sents)),  # number of sentences in the document,
                    'token_count': len(words),
                    'word_count': len(words),
                    'char_count': len(txt),
                    },  # document level statistics
            'by_paragraph': {  # mappings between (item length in THING, count)
                'sentence_length': Counter(para_sen_lens),
                'word_length': Counter(para_word_lens),
                'token_length': Counter(para_tok_lens),
                'char_length': Counter(para_char_lens),
            },
            'by_sentence': {  # mappings between (item length in THING, count)
                'word_length': Counter(sen_word_count),
                'token_length': Counter(sen_tok_count),
                'char_length': Counter(sen_charcount),
            },
            'by_token': {
                'tokens': tokens,  # Mapping (token(str), count)
                # 'token_length': {},  # mappings between (item length in THING, count)
            },
            'by_word': {  # Mapping (word, count)
                'words': Counter(words),  # Mapping (word(str), count)
                # 'word_length': {},  # mappings between (item length in THING, count)
            },

        }
        stats = {'char_count': len(txt),
                 'tokens': {'total_token_count': sum(token_count.values()),
                            'different_token_count': len(list(token_count.keys())),
                            'token_length_stats': token_stats,
                            'total_word_count': len(words),
                            'different_word_count': len(set(words)),
                            },
                 'sentences': {'sentence_count': len(list(doc.sents)),
                               'char_stats': sen_char_stats,
                               'token_stats': sen_token_stats,
                               'word_stats': sen_word_stats,
                               },
                 'paragraphs': para_stats,
                 }
        # stats: aggregated statistics
        # stats_data: all data count
        # tokens: tokens, count of each token and token set
        return metainfo, stats, stats_data, tokens


def process_file(fname, rdf_metadata):
    saveto = fname.replace('.zip', '.stats.tar.gz')
    # if output file exists, ignore as it was already processed
    if os.path.exists(saveto):
        return
    try:
        print("Processing {}".format(fname))
        rfd_metadata, stats, stats_data, tokens = process_gutenberg_file(fname, rdf_metadata)

        outdict = {
            'metadata': rdf_metadata,
            'file_stats': stats,  # statistics
            'stats_data': stats_data,  # statistics that are useful to aggregate with other results
            'tokens': tokens
        }

        jsn = json.dumps(outdict)

        with gzip.open(saveto, 'wb') as f:
            print("Saving to {}".format(saveto))
            f.write(jsn)
            f.flush()

    except Exception as e:  # handling to avoid issues in the starmap pool
        print("Error processing file {} with Exception {}".format(fname, e))


def process_gutenberg(filelist, rfd_meta, n_proc=cpu_count()):
    # pre-process file list and parametrization
    params = []
    for fname in filelist:
        # Gutenberg file id
        gut_id = int(get_file_id(fname))
        # meta information extracted from the Gutenberg RFD database -> warning the DB is a file of about 1GB
        try:
            meta = rfd_meta[gut_id]
            params.append((fname, meta))
        except KeyError as e:
            print("ERROR No Metadata entry for id {} on file {}".format(gut_id, fname))
    with Pool(processes=n_proc) as pool:
        res = pool.starmap(process_file, params)


def _get_filest_from_ziplist(filelist, base_dir):
    flist = [os.path.join(base_dir, f.strip().replace('\n', '').split(' ')[-1]) for f in filelist]
    return flist


def main(zipfilelist=ZIP_FILE_LIST, base_dir=BASE_DIR):
    with open(zipfilelist, "r") as f:
        gutfiles = f.readlines()
        # clean zipfile that contains some garbage
        gutfiles = _get_filest_from_ziplist(gutfiles, base_dir)

    gut_metadata = readmetadata(RDF_TAR_FILE)
    process_gutenberg(gutfiles, gut_metadata, cpu_count()*2)


if __name__ == '__main__':
    main()

