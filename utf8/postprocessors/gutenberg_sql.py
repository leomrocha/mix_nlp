
# start with metainfo
CREATE_METADATA_AUTHOR = """CREATE TABLE IF NOT EXISTS metadata.author (
    name, (unique id
    yearofbirth,
    yearofdeath,
    );
"""
CREATE_METADATA_BOOK = """CREATE TABLE IF NOT EXISTS metadata.book (
    gutid, (unique gutenberg id)
    title,
    author (reference),
    language,
    );
"""
CREATE_METADATA_BOOK_LINKS = """CREATE TABLE IF NOT EXISTS metadata.book_links (
    gutid (reference),
    format,
    url
    );
"""

# general statistics tables
CREATE_STATS_GENERAL_TOKENS = """CREATE TABLE IF NOT EXISTS stats_general.tokens (
 gutid (reference, unique),
 total_token_count
 different_token_count
 total_word_count
 different_word_count
 length_total
 length_min
 length_max
 length_mean
 length_median
 length_std
); 
"""

CREATE_STATS_GENERAL_SENTENCES = """CREATE TABLE IF NOT EXISTS stats_general.sentences (
    gutid (reference, unique),
    sentence_count_total,
    sentence_count_min,
    sentence_count_max,
    sentence_count_mean,
    sentence_count_median,
    sentence_count_std, 
    char_stats_total,
    char_stats_min,
    char_stats_max,
    char_stats_mean,
    char_stats_median,
    char_stats_std, 
    token_stats_total,
    token_stats_min,
    token_stats_max,
    token_stats_mean,
    token_stats_median,
    token_stats_std, 
    word_stats_total,
    word_stats_min,
    word_stats_max,
    word_stats_mean,
    word_stats_median,
    word_stats_std,
); 
"""
CREATE_STATS_GENERAL_PARAGRAPHS = """CREATE TABLE IF NOT EXISTS stats_general.paragraphs (
    gutid (reference, unique),
    length_total,
    length_min,
    length_max,
    length_mean,
    length_median,
    length_std,
    ); 
"""


