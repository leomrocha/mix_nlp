"""

Loading Data


# TODO
Do the Data Loading with NVIDIA DALI instead, examples and documentation here:

https://github.com/NVIDIA/DALI
https://developer.nvidia.com/DALI
https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html
https://github.com/NVIDIA/DALI/tree/master/docs/examples

https://towardsdatascience.com/nvidia-dali-speeding-up-pytorch-876c80182440
https://github.com/yaysummeriscoming/DALI_pytorch_demo

"""

import math
import numpy as np
import orjson as json
import string
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import IterableDataset
import unidecode

try:
    from .constants import *
except:
    # hack to solve issue with ipython executing this import
    from constants import *


# definitions of functions for text alteration at character level
# swap two (consecutive) characters
def swap_consecutive(sentence, idx1, idx2):
    """
    swaps two elements in a sequence, it copies the original array
    :param arr: a numpy array
    :param idx1: The indices to swap
    :param idx2: The indices to swap
    :return: the new array with swapped elements
    """
    arr = np.copy(sentence)
    tmp1, tmp2 = arr[idx1], arr[idx2]
    arr[idx1] = tmp2
    arr[idx2] = tmp1
    return arr, sentence


# mask a set of characters, better if they are close
# add/replace with random character from the current sentence
# add/replace with random character from the entire charset (avoid the special charset space)

# most of this function is taken from fairseq legacy fairseq/data/legacy/masked_lm_dataset.py and modified
def generate_mask(sentence, masking_ratio=0.15, masking_prob=0.8,
                  random_token_prob=0.1, mask_idx=MSK[1],
                  dictionary_token_range=(32, 2112)):
    """
    Generates a mask and returns it, also returning the original sentence
    :param sentence: np.array (int) of the sentence representation
    :param masking_ratio: ratio of the sentence to be masked
    :param masking_prob: masking probability of the
    :param random_token_prob: randomly choose one element in the current sentence to replace with
    :param mask_idx: idx of the masking item in the mapping dictionary
    :param dictionary_token_range: the token range of the symbol mapping dictionary. The first 32 elements are reserved.
    :return:
    """
    masked_sent = np.copy(sentence)
    sent_length = len(sentence)
    mask_num = math.ceil(sent_length * masking_ratio)
    mask = np.random.choice(sent_length, mask_num, replace=False)
    for i in range(sent_length):
        if i in mask:
            rand = np.random.random()
            # replace with mask if probability is less than masking_prob
            # (Eg: 0.8)
            if rand < masking_prob:
                # print(i, mask_idx)
                masked_sent[i] = mask_idx

            # replace with random token if probability is less than
            # masking_prob + random_token_prob (Eg: 0.9)
            elif rand < (masking_prob + random_token_prob):
                # toss a coin to see if should replace with current sentence or all the available symbols
                if np.random.choice([True, False]):
                    # sample random token from CURRENT sentence, no need to complicate it much more in this case
                    masked_sent[i] = np.random.choice(sentence)
                else:
                    masked_sent[i] = (
                        np.random.randint(
                            dictionary_token_range[0], dictionary_token_range[1]
                        )
                    )

    return masked_sent, sentence


# swap characters
# delete character
# duplicate character
# remove diacritics to (a part of the) text
# randomly swap capitalization of a letter  -> this one might be needed to be high prob
def add_str_noise(sentence, dup_char_prob=0.01, del_char_prob=0.005, remove_diacritics_prob=0.2,
                  case_noise_ratio=0.15):
    """
    Generates string noise swapping, adding and deleting characters, changing capitalization and removing diacritics
    :param sentence:
    :param dup_char_prob:
    :param del_char_prob:
    :param remove_diacritics_prob:
    :param case_noise_ratio:
    :return:
    """
    noise_sentence = sentence
    if np.random.random() < remove_diacritics_prob:  # this might be overkill, but should be used
        noise_sentence = unidecode.unidecode(sentence)
    # convert now to list to deal with character level noise addition
    noise_sentence = list(noise_sentence)
    sent_length = len(noise_sentence)
    dup_mask_num = math.ceil(sent_length * dup_char_prob)
    dup_mask = np.random.choice(sent_length, dup_mask_num, replace=False)
    sent_length = len(noise_sentence)
    for i in range(sent_length):
        if i in dup_mask:
            if i > 0:
                noise_sentence = noise_sentence[:i] + noise_sentence[i - 1:]
    # character deletion
    del_mask_num = math.ceil(sent_length * del_char_prob)
    del_mask = np.random.choice(sent_length, del_mask_num, replace=False)
    del_sentence = []
    for i in range(sent_length + 1):
        if i in del_mask and i < sent_length - 1:
            continue
        else:
            del_sentence.append(noise_sentence[i])
    noise_sentence = del_sentence

    sent_length = len(noise_sentence)
    # Casing is important
    # 1. to show the network the relationship between lower and uppercase (this is not pre-filtered)
    # 2. For languages that depend on casing
    mask_num = math.ceil(sent_length * case_noise_ratio)
    case_mask = np.random.choice(sent_length, mask_num, replace=False)

    for i in range(sent_length):
        if i in case_mask:
            noise_sentence[i] = noise_sentence[i].swapcase()

    return noise_sentence, sentence


def code2str(code, int2char):
    """
    Remaps the given numpy array (a coded string) back to the represented string
    :param code: numpy 1D array
    :param int2char: mapping
    :return: the string represented by the array
    """
    return ''.join([int2char[i] for i in code])


class Txt2TxtDataset(IterableDataset):
    def __init__(self, root_dir, codedict,
                 dup_char_prob=0.01, del_char_prob=0.005, remove_diacritics_prob=0.2, case_noise_ratio=0.15,
                 masking_ratio=0.15, masking_prob=0.8, random_token_prob=0.1,
                 separator=SEPARATOR, special_codes=SPECIAL_CODES,
                 nil=NUL, soh=SOH, stx=STX, etx=ETX, eot=EOT, unk=UNK, msk=MSK,
                 ):
        self.root_dir = root_dir
        self.separator = separator
        self.codedict = codedict
        self.special_codes = special_codes
        # verify that the special codes are present
        self.unk = unk
        self.pad = nil
        self.start_header = soh
        self.start_text = stx
        self.end_text = etx
        self.end_transaction = eot
        self.mask = msk

        for c in special_codes:
            if c[0] not in self.codedict:
                self.codedict[c[0]] = c[1]

        self.dup_char_prob = dup_char_prob
        self.del_char_prob = del_char_prob
        self.remove_diacritics_prob = remove_diacritics_prob
        self.case_noise_ratio = case_noise_ratio
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob

        # TODO the reserved space is to be given by conf
        self.dictionary_token_range = (32, max(codedict.values()))

    def _txt2tensor(self, txt):
        return np.array(list(map(self._item2int, txt.decode('utf-8'))))

    def _item2int(self, char):
        if char not in self.codedict:
            char = self.unk[1]
        num = self.codedict[char]
        return num

    def _form_langmodel_pair(self, sentence):
        # noise addition
        noised_sentence, sentence = add_str_noise(sentence, self.dup_char_prob, self.del_char_prob,
                                                  self.remove_diacritics_prob,
                                                  self.case_noise_ratio)
        # encode
        noised_code = self._txt2tensor(noised_sentence)
        sentence_code = self._txt2tensor(sentence)
        # mask addition
        masked_sentence, _ = generate_mask(noised_code, self.masking_ratio, self.masking_prob,
                                                  self.random_token_prob, self.mask[1],
                                                  self.dictionary_token_range)
        # WARNING these still need to be padded but not in this function
        # Now add the start and end of text tags and the end of transaction tag
        start_tag = self.start_text[1]
        end_tags = [self.end_text[1], self.end_transaction[1]]
        masked_ret = np.zeros(masked_sentence.shape[0] + 3)
        masked_ret[0] = start_tag
        masked_ret[-2:] = end_tags
        masked_ret[1:-2] = masked_sentence

        sentence_ret = np.zeros(sentence_code.shape[0] + 3)
        sentence_ret[0] = start_tag
        sentence_ret[-2:] = end_tags
        sentence_ret[1:-2] = sentence_code

        return masked_ret, sentence_ret

    def _form_task_tuple(self, task_txt, src_txt, src_lang, tgt_txt, add_noise=False):
        """
        Form the arrays for
        :param task_txt:
        :param src_txt:
        :param src_lang:
        :param tgt_txt:
        :param add_noise: if noise should be added instead
        :return:
        """
        # should it be corrupted or not, should it return the original plus the corrupted?
        start_tag = self.start_header[1]
        start_txt_tag = self.start_text[1]
        end_txt_tag = self.end_text[1]
        end_tx_tag = self.end_transaction[1]
        # WARNING these still need to be padded later

        lang = ''.join([start_txt_tag, src_lang, end_txt_tag])
        target_lang = self._txt2tensor(lang)
        target = self._txt2tensor(''.join([start_txt_tag, tgt_txt, end_txt_tag, end_tx_tag]))

        if add_noise:
            noise_masked, noise_sentence = self._form_langmodel_pair(src_txt)
            # as both noise and no noise are returned there is another extra training point with the same input
            return noise_masked, noise_sentence, target_lang, target
        # else
        txt = ''.join([start_tag, task_txt, start_txt_tag, src_txt, end_txt_tag, end_tx_tag] )
        source = self._txt2tensor(txt)
        return source, target_lang, target

    def process_line(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        # TODO the parsing here should be of json instead ...
        # src_txt, src_lang, tgt_txt = line.split(self.separator)
        # corrupt the sentence for lang model:
        # TODO
        # src_txt = self._txt2tensor(src_txt)
        # src_lang = self._txt2tensor(src_lang)
        # tgt_txt = self._txt2tensor(tgt_txt)
        # TODO pad to dimension!
        pass
        # return src_txt, src_lang, tgt_txt

    def __iter__(self):
        # TODO this should be better for reading big files ...
        iterator = open(self.root_dir, 'rb')
        ret = map(self.process_line, iterator)
        return ret

    # def __next__(self):
    #     # .... ???
    #     if :
    #         return x
    #     else:
    #         raise StopIteration
