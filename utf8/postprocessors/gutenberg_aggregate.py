#!/usr/bin/env python3

import os
import sys
import gzip
import pandas as pd
from pathlib import Path

try:
    import orjson as json
    # import ujson as json
except:
    import json
import sqlite3


def get_path_and_dict(stats):
    first_value = next(iter(mydict.values()))
    if type(first_value) == int:
        pass



def sum_stats(data_source_path):
    '''
    :param data_source_path: directory with json.gz files to aggregate
    :returns: a dict with summed values
    '''
    
    # return {'list': os.listdir(data_source_path)}
    aggregated_stats = {
        'doc': pd.Series(dtype=int), 
        'by_paragraph': {
            'sentence_length': pd.Series(dtype=int), 
            'word_length': pd.Series(dtype=int), 
            'token_length': pd.Series(dtype=int), 
            'char_length': pd.Series(dtype=int),
        }, 
        'by_sentence': { 
            'word_length': pd.Series(dtype=int), 
            'token_length': pd.Series(dtype=int), 
            'char_length': pd.Series(dtype=int),
        }, 
        # uncomment this to include words statistics, 
        # will result in larger files (do not commit them) and 
        # the computation will be slower 
        # (as pandas does some aligning on the index https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html?highlight=alignment#vectorized-operations-and-label-alignment-with-series)
        # 'by_word': {
        #     'words': pd.Series(dtype=int)
        # }
    }

    # get all files (including sub directories)
    paths = list(Path(data_source_path).rglob("*stats_all*.json.gz"))


    levels = list(aggregated_stats.keys())
    levels.remove('doc')
    # sum up the stats
    for path in paths:
        with gzip.open(path) as f:
            book = json.loads(f.read())
        
        stats = book['stats_data']
        
        aggregated_stats['doc'] = aggregated_stats['doc'].add(
            pd.Series(stats['doc']), fill_value=0
        ).astype(int)

        # MAYBE: deal with nesting dynamically
        for level in levels:
            for key in aggregated_stats[level].keys():
                aggregated_stats[level][key] = aggregated_stats[level][key].add(
                    pd.Series(stats[level][key]), fill_value=0
                ).astype(int)

    # remove pandas
    aggregated_stats['doc'] = aggregated_stats['doc'].to_dict()
    for level in levels:
        for key, value in aggregated_stats[level].items():
            aggregated_stats[level][key] = value.to_dict()

    return aggregated_stats


def aggregate_sample():
    sample_data_path = 'gutenberg_stats_examples'
    # print(sum_stats(sample_data_path)['doc'])
    print(sum_stats(sample_data_path))


def aggregate_by_language():
    languages_dir = 'gutenberg_stats/clean_results'
    languages = os.listdir(languages_dir)

    for language in languages:
        print(f'Aggregating by language: [{language}]')
        aggregated_stats = sum_stats(f'{languages_dir}/{language}')
        json_data = json.dumps(aggregated_stats)

        target_dir = f'gutenberg_agg_stats/by_language'
        os.makedirs(target_dir, exist_ok=True)

        with open(f'{target_dir}/{language}.json', 'w') as f:
            f.write(json_data)


def create_db():
    # TODO, all the rest
    #  for the moment I'm just starting to structure the DB ideas in the gutenberg_sql.py file
    pass



def main():
    # aggregate_sample()
    aggregate_by_language()


if __name__ == '__main__':
    main()
