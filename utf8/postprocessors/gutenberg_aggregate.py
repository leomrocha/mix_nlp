#!/usr/bin/env python3

import os
import sys
import gzip
from pathlib import Path
from collections import Counter

try:
    import orjson as json
    # import ujson as json
except:
    import json
import sqlite3


LANGUAGES_DIR = 'gutenberg_stats/clean_results'
TARGET_DIR = 'gutenberg_stats_short/aggregate'

# uncomment to include by word statistics, will not be committed
# TARGET_DIR = 'gutenberg_stats/aggregate'


def main():
    # aggregate_sample()
    aggregate_by_language()

    if not 'short' in TARGET_DIR:
        aggregate_by_author()


def sum_stats(data_source_path):
    '''
    :param data_source_path: directory with json.gz files to aggregate
    :returns: a dict with summed values
    '''
    
    aggregated_stats = {
        'doc': Counter(), 
        'by_paragraph': {
            'sentence_length': Counter(), 
            'word_length': Counter(), 
            'token_length': Counter(), 
            'char_length': Counter(),
        }, 
        'by_sentence': { 
            'word_length': Counter(), 
            'token_length': Counter(), 
            'char_length': Counter(),
        }, 
    }

    # will result in larger files and the computation will be slower 
    if not 'short' in TARGET_DIR:
        aggregated_stats['by_word'] = {'words': Counter()}

    # get all files (including sub directories)
    paths = list(Path(data_source_path).rglob("*stats_all*.json.gz"))

    levels = list(aggregated_stats.keys())
    levels.remove('doc')
    # sum up the stats
    for path in paths:
        with gzip.open(path) as f:
            book = json.loads(f.read())
        
        book_stats = book['stats_data']
        
        aggregated_stats['doc'].update(book_stats['doc'])

        # MAYBE: deal with nesting dynamically
        for level in levels:
            for dist_name, agg_dist in aggregated_stats[level].items():
                agg_dist.update(book_stats[level][dist_name])

    # MAYBE: remove Counters

    return aggregated_stats


# for manual testing purposes
def aggregate_sample():
    sample_data_path = 'gutenberg_stats_examples'
    # print(sum_stats(sample_data_path)['doc'])
    print(sum_stats(sample_data_path))


def aggregate_by_language():
    languages = os.listdir(LANGUAGES_DIR)

    for language in languages:
        print(f'Aggregating by language: [{language}]')
        aggregated_stats = sum_stats(f'{LANGUAGES_DIR}/{language}')
        json_data = json.dumps(aggregated_stats)

        target_dir = f'{TARGET_DIR}/by_language'
        os.makedirs(target_dir, exist_ok=True)

        with open(f'{target_dir}/{language}.json', 'w') as f:
            f.write(json_data)


def aggregate_by_author():
    languages = os.listdir(LANGUAGES_DIR)

    for language in languages:

        authors = os.listdir(f'{LANGUAGES_DIR}/{language}')

        for author in authors:

            print(f'Aggregating by language: [{language}], Author: [{author}]')
            aggregated_stats = sum_stats(f'{LANGUAGES_DIR}/{language}/{author}')
            json_data = json.dumps(aggregated_stats)

            target_dir = f'{TARGET_DIR}/by_author/{language}/'
            os.makedirs(target_dir, exist_ok=True)

            with open(f'{target_dir}/{author}.json', 'w') as f:
                f.write(json_data)


def create_db():
    # TODO, all the rest
    #  for the moment I'm just starting to structure the DB ideas in the gutenberg_sql.py file
    pass


if __name__ == '__main__':
    main()
