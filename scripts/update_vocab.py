# Update vocab
import os
import sys
import json
import numpy as np
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, BertTokenizer


def read_json(filepath):
    with open(filepath, 'r') as fi:
        data = json.load(fi)
    return data

def check_unused():
    special_tokens = {'[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]'}
    def fn(word_tuple):
        word, idx = word_tuple
        if word in special_tokens:
            return False
        word = word.lower()
        for ch in word:
            is_cn = '\u4e00' <= ch <= '\u9fff'
            is_en = 'a' <= ch <= 'z'
            is_num = '0' <= ch <= '9'
            is_unused = is_cn or is_en or is_num
            if not is_unused:
                return True
        return False
    return fn

def main(sentences, vocab):
    '''将vocab词表里面没有使用过的词替换成新词
    sentences: list of string
    vocab: OrderedDict, key[string], value[integer]
    '''
    idx = 0
    total_tokens = set([word for sentence in sentences for word in sentence if word.strip()])
    new_tokens = list(filter(lambda w: w not in vocab, total_tokens))
    new_vocab = list(vocab.items())
    fn = check_unused()
    unused_tokens = list(filter(fn, vocab.items()))
    print('Unused tokens: {}, new tokens: {}'.format(len(unused_tokens), len(new_tokens)))
    if len(unused_tokens) < len(new_tokens):
        new_tokens = new_tokens[:len(unused_tokens)]
    for token in new_tokens:
        token_idx = unused_tokens[idx][1]
        new_vocab[token_idx] = [token, token_idx]
        idx += 1

    return OrderedDict(new_vocab)


if __name__ == '__main__':

    data1 = read_json('./data/xueqiu.json')
    data2 = read_json('./data/sina.json')
    sentences = list(map(lambda x: x['title'], data1+data2))

    model_dir = './multilingual_sentiment_vocab20k'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    vocab = tokenizer.vocab

    new_vocab = main(sentences, vocab)
    tokenizer.vocab = tokenizer
    # tokenizer.save_pretrained(model_dir)
