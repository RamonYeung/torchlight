"""
Utilizations for common usages.
"""
import os
import random
import torch
from difflib import SequenceMatcher
from unidecode import unidecode
from datetime import datetime


def personal_display_settings():
    """
    Pandas Doc
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html
    NumPy Doc
        -
    """
    from pandas import set_option
    set_option('display.max_rows', 500)
    set_option('display.max_columns', 500)
    set_option('display.width', 2000)
    set_option('display.max_colwidth', 1000)
    from numpy import set_printoptions
    set_printoptions(suppress=True)


def set_seed(seed):
    """
    Freeze every seed.
    All about reproducibility
    TODO multiple GPU seed, torch.cuda.all_seed()
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def normalize(s):
    """
    German and Frence have different vowels than English.
    This utilization removes all the non-unicode characters.
    Example:
        āáǎà  -->  aaaa
        ōóǒò  -->  oooo
        ēéěè  -->  eeee
        īíǐì  -->  iiii
        ūúǔù  -->  uuuu
        ǖǘǚǜ  -->  uuuu

    :param s: unicode string
    :return:  unicode string with regular English characters.
    """
    s = s.strip().lower()
    s = unidecode(s)
    return s


def snapshot(model, epoch, save_path):
    """
    Saving model w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    os.makedirs(save_path, exist_ok=True)
    current = datetime.now()
    timestamp = f'{current.month:02d}{current.day:02d}_{current.hour:02d}{current.minute:02d}'
    torch.save(model.state_dict(), save_path + f'{type(model).__name__}_{timestamp}_{epoch}th_epoch.pkl')


def show_params(model):
    """
    Show model parameters for logging.
    """
    for name, param in model.named_parameters():
        print('%-16s' % name, param.size())


def longest_substring(str1, str2):
    # initialize SequenceMatcher object with input string
    seqMatch = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    # print longest substring
    return str1[match.a: match.a + match.size] if match.size != 0 else ""


def pad(sent, max_len):
    """
    syntax "[0] * int" only works properly for Python 3.5+
    Note that in testing time, the length of a sentence
    might exceed the pre-defined max_len (of training data).
    """
    length = len(sent)
    return (sent + [0] * (max_len - length))[:max_len] if length < max_len else sent[:max_len]


def to_cuda(*args, device=None):
    """
    Move Tensors to CUDA. 
    If no device provided, default to the first card in CUDA_VISIBLE_DEVICES.
    """
    assert all(torch.is_tensor(t) for t in args), \
            'Only support for tensors, please check if any nn.Module exists.'
    if device is None:
        device = torch.device('cuda:0')
    return [None if x is None else x.to(device) for x in args]


if __name__ == '__main__':
    print(normalize('ǖǘǚǜ'))
