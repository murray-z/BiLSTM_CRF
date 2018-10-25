# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/21 9:54
@Author  : fazhanzhang
@Function :
"""

config = {
    'learning_rate': 0.03,
    'vocab_size': 5000,
    'embedding_size': 128,
    'hidden_size': 128,
    'dropout_keep_prob': 0.5,
    'clip_grad': 5,
    'max_model_keep': 5,
    'data_path': './data/ner.txt',
    'batch_size': 1,
    'num_epochs': 1,
    'evaluate_every': 100,
    'checkpoint_every': 100,
    'vocab_path': './vocabs',
    'train_test_dev_rate': [0.98, 0.01, 0.01]
}