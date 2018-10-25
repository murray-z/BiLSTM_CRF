# -*- coding: utf-8 -*-

"""
@Time    : 2018/7/29 10:18
@Author  : fazhanzhang
@Function :
"""

import tensorflow as tf


class BilstmCrf(object):
    def __init__(self, config):
        self.learning_rate = config['learning_rate']
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_tags = config['num_tags']

        # placeholder
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_y')
        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

        # embedding
        with tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_normal([self.vocab_size, self.embedding_size], -1.0, 1.0, name='W')
            )
            self.embedding = tf.nn.embedding_lookup(self.W, self.input_x, name='embedding')

        # bi-lstm
        with tf.name_scope('bi-lstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32
            )
            self.output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            self.output = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # output layer
        with tf.name_scope('output'):
            W = tf.get_variable(name='W',
                                shape=[2*self.hidden_size, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            s = tf.shape(self.output)
            output = tf.reshape(self.output, [-1, 2*self.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

        # loss
        with tf.name_scope('loss'):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.input_y,
                sequence_lengths=self.sequence_length
            )
            self.loss = -tf.reduce_mean(log_likelihood)


if __name__ == '__main__':
    pass