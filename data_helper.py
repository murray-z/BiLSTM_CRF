# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/21 10:42
@Author  : fazhanzhang
@Function :
"""

from config import config
from collections import Counter
import os
import json
import numpy as np
import sys


def extract_ner_from_raw(input_file_path, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f_w:
        with open(input_file_path, 'r', encoding='utf-8') as f_r:
            for line in f_r.readlines():
                lis = line.strip().split('\t')
                string = lis[0]
                ner_tag = []
                for item in lis[1].split():
                    if item.split('-')[1] in ['PER', 'ORG', 'TIME', 'LOC']:
                        ner_tag.append(item)
                    else:
                        ner_tag.append('O')
                f_w.write('{}\t{}\n'.format(string, ' '.join(ner_tag)))


def process_data(config):
    """
    文本格式
    :param config:
    :return:
    """
    data_path = config['data_path']
    X = []
    y = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            lis = line.strip().split('\t')
            if 'B' in lis[-1]:

                X.append(lis[0])
                y.append(lis[1])
    return X, y


def generate_vocab(X, y, config):
    words = []
    for sent in X:
        words.extend(list(sent))
    words = Counter(words).most_common(config['vocab_size']-1)

    word_to_index = {}
    index_to_word = {}
    for i in range(len(words)):
        word_to_index[words[i][0]] = i+1
        index_to_word[int(i+1)] = words[i][0]

    word_to_index['UNK'] = 0
    index_to_word[int(0)] = 'UNK'

    label_to_index = {}
    index_to_label = {}
    labels = set([tag for str_tags in y for tag in str_tags.split()])

    idx = 1
    for label in labels:
        if label != 'O':
            label_to_index[label] = idx
            index_to_label[idx] = label
            idx += 1
        else:
            label_to_index['O'] = 0
            index_to_label[0] = 'O'

    vocab_path = config['vocab_path']

    if not os.path.exists(vocab_path):
        os.mkdir(vocab_path)

    with open(os.path.join(vocab_path, 'word_to_index.json'), 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f, ensure_ascii=False)

    with open(os.path.join(vocab_path, 'index_to_word.json'), 'w', encoding='utf-8') as f:
        json.dump(index_to_word, f, ensure_ascii=False)

    with open(os.path.join(vocab_path, 'label_to_index.json'), 'w', encoding='utf-8') as f:
        json.dump(label_to_index, f, ensure_ascii=False)

    with open(os.path.join(vocab_path, 'index_to_label.json'), 'w', encoding='utf-8') as f:
        json.dump(index_to_label, f, ensure_ascii=False)

    return word_to_index, label_to_index


def padding(X, y, word_to_index, label_to_index):
    sequence_lengths = []
    max_sequence_length = config['max_seq_length']
    input_x = []
    for line in X:
        temp = []
        for item in list(line):
            temp.append(word_to_index.get(item, 0))
        sequence_lengths.append(min(len(temp), max_sequence_length))
        input_x.append(temp[:max_sequence_length]+[0]*(max_sequence_length-len(temp)))
    if not y:
        return input_x

    input_y = []
    for str_tag in y:
        temp = []
        tags = str_tag.split()
        for item in tags:
            temp.append(label_to_index[item])
        input_y.append(temp[:max_sequence_length]+[label_to_index['O']]*(max_sequence_length-len(temp)))
    return input_x, input_y, sequence_lengths


def split_data(input_x, input_y, sequence_lengths, config):
    rate = config['train_test_dev_rate']
    shuffle_indices = np.random.permutation(np.arange(len(input_y)))
    # print(shuffle_indices)
    # print(input_y)
    x_shuffled = np.array(input_x)[shuffle_indices]
    y_shuffled = np.array(input_y)[shuffle_indices]
    sequence_lengths = np.array(sequence_lengths)[shuffle_indices]

    x_train, y_train, sequences_length_train = x_shuffled[: int(rate[0]*len(input_y))], \
                                               y_shuffled[: int(rate[0]*len(input_y))], \
                                               sequence_lengths[: int(rate[0]*len(input_y))]

    x_test, y_test, sequence_length_test = x_shuffled[int(rate[0]*len(input_y)): int(sum(rate[:2])*len(input_y))], \
                     y_shuffled[int(rate[0]*len(input_y)): int(sum(rate[:2])*len(input_y))], \
                     sequence_lengths[int(rate[0]*len(input_y)): int(sum(rate[:2])*len(input_y))]

    x_dev, y_dev, sequence_length_dev = x_shuffled[int(sum(rate[:2])*len(input_y)):],\
                                        y_shuffled[int(sum(rate[:2])*len(input_y)):], \
                                        sequence_lengths[int(sum(rate[:2])*len(input_y)):]
    return x_train, y_train, sequences_length_train, x_test, y_test, sequence_length_test, x_dev, y_dev, sequence_length_dev


def generate_batchs(x_train, y_train, sequence_length_train, config, shuffle=True):
    data = np.array(list(zip(x_train, y_train, sequence_length_train)))
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/config['batch_size']) + 1
    for epoch in range(config['num_epochs']):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * config['batch_size']
            end_index = min((batch_num+1)*config['batch_size'], data_size)
            yield shuffle_data[start_index: end_index]


def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read(), encoding='utf-8')


def measure(labels, logits, seq_len_list):
    """
    计算acc,recall,f1
    :param labels:
    :param logits:
    :param seq_len_list:
    :return:
    """
    index_to_label = load_json('./vocabs/index_to_label.json')

    accuracy = [0.0, 0.0]
    recall = [0.0, 0.0]
    all_labels = []
    all_logits = []
    for i in range(len(labels)):
        len_seq = seq_len_list[i]
        all_labels.extend(list(labels[i])[:len_seq])
        all_logits.extend(list(logits[i])[:len_seq])

    if len(all_logits) != len(all_labels):
        print('len_labels != len_logits')
        sys.exit()

    all_labels = [index_to_label[str(item)] for item in all_labels]
    all_logits = [index_to_label[str(item)] for item in all_logits]

    # print(all_labels)
    # print(all_logits)


    flag = False
    tag = ''
    for i in range(len(all_labels)):
        if not flag and all_labels[i].startswith('B'):
            tag = all_labels[i].split('-')[1]
            header = i
            flag = True
            recall[1] += 1

        if flag and tag not in all_labels[i]:
            if all_labels[header] == all_logits[header] and all_labels[i-1] == all_logits[i-1]:
                recall[0] += 1
                flag = False

    flag = False
    tag = ''
    for i in range(len(all_labels)):
        if not flag and all_logits[i].startswith('B'):
            tag = all_labels[i].split('-')[-1]
            header = i
            flag = True
            accuracy[1] += 1

        if flag and tag not in all_logits[i]:
            if all_labels[header] == all_logits[header] and all_labels[i - 1] == all_logits[i - 1]:
                accuracy[0] += 1
                flag = False

    if accuracy[1] == 0:
        acc = 0.0
    else:
        acc = accuracy[0]/accuracy[1]

    if recall[1] == 0:
        recall = 0
    else:
        recall = recall[0]/recall[1]

    if acc+recall == 0:
        f1 = 0.0
    else:
        f1 = 2*acc*recall/(acc+recall)

    return acc, recall, f1


if __name__ == '__main__':
    labels = [[0, 2, 4, 4, 0, 0, 6, 5, 0]]
    logits = [[0, 2, 4, 4, 0, 6, 5, 5, 0]]
    print(measure(labels, logits, [9]))