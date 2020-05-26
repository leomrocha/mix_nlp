# general imports for file manipulation
import os
import sys
import zipfile  # to read the zipped gutenberg text files
import rdflib  # to read the rdf directory of the gutenberg files
from pathlib import Path  # to deal with file paths, naming and other things in a platform independent manner
import numpy as np
# NLP imports
import spacy
# language imports
from pycountry import languages  # to deal with language naming conventions
# multiprocessing imports
from multiprocessing import Pool, cpu_count

# local tools imports
# gutenberg project metadata parsing
try:
    from .gutenberg_metainfo import *  # slightly modified script so it works for what I need
except:
    from gutenberg_metainfo import *  # slightly modified script so it works for what I need

# gutenberg specific imports
import gutenberg_cleaner  # to clean headers and footers from gutenberg files, they are NOISE

BASE_DIR = "/media/nfs/Datasets/text/Gutenberg/aleph.gutenberg.org"
RDF_TAR_FILE = "/media/nfs/Datasets/text/Gutenberg/rdf-files.tar.bz2"
# BASE_RDF_DIR = "/media/nfs/Datasets/text/Gutenberg/rdf_db/cache/epub"

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
    nlp = spacy.load(lang)
    return


def process_gutenberg(filelist):
    with Pool(processes=cpu_count()) as pool:
        pass
        # res = pool.map(extract_charset, all_files)


# get next file

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

def process_file(fname, rfd_meta):
    # Gutenberg file id
    gut_id = int(get_file_id(fname))
    # meta information extracted from the Gutenber RFD database -> warning is around 1GB the DB
    metainfo = rfd_meta[gut_id]
    # TODO extract format from metadata (if exists)
    encoding = 'utf-8'  # asume all is compatible with utf-8, for the moment haven't found one that is not
    # spacy
    # nlp = get_nlp_resource(metainfo)
    # TODO FIXME, this is an issue with current spacy install not loading correctly ??
    import en_core_web_md
    nlp = en_core_web_md.load()
    # load and clean the file, asume zip as compressing format
    pth = Path(fname)  #
    ftxt = pth.name.replace(".zip", ".txt")  # inside gutenberg zip there should be a .txt file with the same name
    f = zipfile.ZipFile(fname)
    txt = f.read(ftxt).decode(encoding)
    txt = gutenberg_cleaner.simple_cleaner(txt)
    # Start analysis
    doc = nlp(txt)  # SpaCy tokenization

    tokens = {}

    stats_data = {'total_char_count': len(txt)}
    #     stats['total_token_len'] = len(list(doc))  # can do it in another way and improve procesing speed

    token_count = {}  # individual token count, frequency
    token_lens = []  # to be able to draw histograms of token lengths
    token_set = set()  # all tokens

    tok_stats = {}

    sen_charcount = []  # sentence length in characters
    sen_tok_count = []  # sentence length in tokens
    sen_stats = {
        'sentence_count': len(list(doc.sents))  # number of sentences in the document
    }

    for s in doc.sents:
        clen = len(s.string)
        sen_charcount.append(clen)
        tlen = len(s)
        sen_tok_count.append(tlen)
        # token level
        for t in s:
            # WARNING -> something here might be better done in another way
            token_set.add(t)
            token_lens.append(len(t))
            # FIXME: this is NOT working, every token appears only once
            if t not in token_count:
                token_count[t] = 1
            else:
                token_count[t] += 1

    sen_stats['char_count'] = sen_charcount
    sen_stats['token_count'] = sen_tok_count

    sen_charcount = np.array(sen_charcount)
    sen_tok_count = np.array(sen_tok_count)
    # np.min(), np.max(), np.mean(), np.median(), np.std()
    sen_stats['char_stats'] = {'min': np.min(sen_charcount),
                               'max': np.max(sen_charcount),
                               'mean': np.mean(sen_charcount),
                               'median': np.median(sen_charcount),
                               'std': np.std(sen_charcount)
                               }
    sen_stats['token_stats'] = {'min': np.min(sen_tok_count),
                                'max': np.max(sen_tok_count),
                                'mean': np.mean(sen_tok_count),
                                'median': np.median(sen_tok_count),
                                'std': np.std(sen_tok_count)
                                }

    stats_data['sentences'] = sen_stats
    stats_data['token_lengths'] = token_lens

    tokens['token_count'] = token_count
    tokens['token_set'] = token_set

    # slows a lot the processing, but I don't care for the moment
    # I'm doubting using this as much of it is already done in the previous part
    # also makes everything slooower ... need to do it faster so I wont count this stat (even though I do want it)
    # i might be able to do some estimation?, nevertheless
    # count number of paragraphs .. might not work on many languages
    paragraphs = [l.strip() for l in txt.split('\n\n') if len(l.strip()) > 0]

    para_char_lens = []
    para_tok_lens = []
    para_sen_lens = []
    for p in paragraphs:
        d = nlp(p)
        para_char_lens.append(len(p))
        para_tok_lens.append(len(list(d)))
        para_sen_lens.append(len(list(d.sents)))
    #
    stats_data['paragraphs'] = {
        'paragraph_count': len(paragraphs),
        'char_count': para_char_lens,
        'token_count': para_tok_lens,
        'sentence_count': para_sen_lens,
    }

    para_char_lens = np.array(para_char_lens)
    para_tok_lens = np.array(para_tok_lens)
    para_sen_lens = np.array(para_sen_lens)

    main_para_char_stats = {'min': np.min(para_char_lens),
                            'max': np.max(para_char_lens),
                            'mean': np.mean(para_char_lens),
                            'median': np.median(para_char_lens),
                            'std': np.std(para_char_lens)
                            }
    main_para_tok_stats = {'min': np.min(para_tok_lens),
                           'max': np.max(para_tok_lens),
                           'mean': np.mean(para_tok_lens),
                           'median': np.median(para_tok_lens),
                           'std': np.std(para_tok_lens)
                           }
    main_para_sen_stats = {'min': np.min(para_sen_lens),
                           'max': np.max(para_sen_lens),
                           'mean': np.mean(para_sen_lens),
                           'median': np.median(para_sen_lens),
                           'std': np.std(para_sen_lens)
                           }
    para_stats = {'char_stats': main_para_char_stats,
                  'token_stats': main_para_tok_stats,
                  'sentence_stats': main_para_sen_stats}

    stats = {'char_count': len(txt),
             'sentence_count': len(list(doc.sents)),
             'total_token_count': sum(token_count.values()),  # something wrong with this value
             'different_token_count': len(token_set),
             'sentence_char_stats': sen_stats['char_stats'],
             'sentence_token_stats': sen_stats['token_stats'],
             'paragraphs': para_stats,
             }
    # stats: aggregated statistics
    # stats_data: all data count
    # tokens: tokens, count of each token and token set
    return stats, stats_data, tokens
