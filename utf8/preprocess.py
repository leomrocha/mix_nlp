import numpy as np
import torch
import torchtext
import torchtext.datasets as ttdatasets

# GLUE

# CoLA

# Description of CoLA tasks and language the format is as follows:
# "2 char LANG code": [descr1, descr2, ... ]
CoLA_TASK_DESC = {
    "en": [""],
    "es": [""],
    "fr": [""],
    "de": [""],
}


def cola_txt2txt(fpath):
    """
    CoLA text definition:
    Each line in the .tsv files consists of 4 tab-separated columns.
    Column 1:	the code representing the source of the sentence.
    Column 2:	the acceptability judgment label (0=unacceptable, 1=acceptable).
    Column 3:	the acceptability judgment as originally notated by the author.
    Column 4:	the sentence.
    :param fpath: input path of the text file to process
    :return: a json description of the task, per each line in the input:
    { 'input': ....
      'target': [unacceptable|acceptable]
    }
    """
    lines = []
    with open(fpath) as f:
        # read line, get only elements 2 and 4,
        # translate 2 to a string value
        # form json
        # append json to lines
        pass
    return lines