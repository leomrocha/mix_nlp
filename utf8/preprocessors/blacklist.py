
# list of blacklisted languages from the current research, this is due to resources availability only.
# languages are chosen by: number of needed codepoints, or by usage to reduce the number of input languages and files
# in the training set and is mostly arbitrary,
MAYBE_BLACKLIST_LANGS = ['ceb', 'jv', 'ce', 'cv', 'dv', 'ht', 'hy', 'ku', 'mh', 'mi', 'ps', 'su', 'tk', 'ba', 'tg',
                         'tt', 'ug'
                         ]
# blacklisting due to lack of samples, extinct language or other issue
EXTRA_BLACKLIST = ["olo", "swl", "bxr", "fa", "sme", "aii", "gun", "yo", "akk", "fo", "mdf",
                   "krl", "pcm", "bho", "sms", "am", "bm", "got", "cu", "hsb", "wo"
                   ]

BLACKLIST_LANGS = ['ar', 'as', 'arz', 'azb', 'bn', 'bp', 'ckb', 'eo', 'ew', 'fa', 'fo', 'gom', 'gu', 'hi', 'hu', 'id',
                   'ilo', 'ja', 'ka', 'kk', 'ko', 'lmo', 'ml', 'mr', 'mwl', 'ne', 'pa', 'py', 'sh', 'si', 'ta', 'te',
                   'th', 'tl', 'ur', 'vi',
                   'wuu', 'yi', 'zb', 'zh'
                   ] + MAYBE_BLACKLIST_LANGS + EXTRA_BLACKLIST

# blacklists to reduce even more the number of languages, latin is left because will be used
# FIRST to train to set a learning baseline ...
# base, greek, old french and ancient greek will be nice too if I manage to transliterate it
# extra blacklisting to reduce the number of alphabets used and other low resource and other non-official EU languages
MORE_EXTRA_BLACKLIST = ["af", "gsw", "he",
                        "ca", "cy", "eu", "ga", "gd", "gl", "cr", "hy", "tr",
                       ]

ANCIENT_LANGS = ["la", "grc", "fro"]  # latin, ancient greek, old french -> base for MANY languages
GREEK_BLACKLIST = ["el", "grc"]

# this means mainly taking out cyrillic scripts ... but bulgarian IS in the EU and uses
# cyrillic so ... what should I do?
CYRILLIC_BLACKLIST = ["be", "bg", "he", "ru", "sr", "uk"]


BLACKLIST_LANGS = BLACKLIST_LANGS + MORE_EXTRA_BLACKLIST + CYRILLIC_BLACKLIST + GREEK_BLACKLIST

BLACKLIST_LANGS = sorted(list(set(BLACKLIST_LANGS)))

