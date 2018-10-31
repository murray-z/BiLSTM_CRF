# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/21 10:42
@Author  : fazhanzhang
@Function :
"""

import tensorflow as tf
from bilstm_crf import BilstmCrf
import time
import os
import datetime
import data_helper
from config import config
import json

if config['use_gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train(config):
    learning_rate = config['learning_rate']
    clip_grad = config['clip_grad']
    max_model_keep = config['max_model_keep']

    print('parameters: ')
    print(json.dumps(config, indent=4, ensure_ascii=False))

    # load data
    print('load data .....')
    X, y = data_helper.process_data(config)

    # make vocab
    print('make vocab .....')
    word_to_index, label_to_index = data_helper.generate_vocab(X, y, config)
    config['num_tags'] = len(label_to_index)

    # padding data
    print('padding data .....')
    input_x, input_y, sequence_lengths = data_helper.padding(X, y, word_to_index, label_to_index)

    # split data
    print('split data .....')
    x_train, y_train, sequences_length_train, x_test, y_test, sequence_length_test, x_dev, y_dev, sequence_length_dev = \
        data_helper.split_data(input_x, input_y, sequence_lengths, config)

    print('length train: {}'.format(len(x_train)))
    print('length test: {}'.format(len(x_test)))
    print('length dev: {}'.format(len(x_dev)))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            bilstm_crf = BilstmCrf(config)

            # training_procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # apply grad clip to avoid gradiend explosion
            grads_and_vars = optimizer.compute_gradients(bilstm_crf.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=global_step)

            # output dir for models and summaries
            timestamp = str(int(time.time()))
            outdir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
            print('writing to {} !!!'.format(outdir))

            # summary of loss
            tf.summary.scalar('loss', bilstm_crf.loss)

            # train summary
            train_sumary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(outdir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev summary
            dev_summary_op = tf.summary.merge_all()
            dev_summary_dir = os.path.join(outdir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint dir
            checkpoint_dir = os.path.abspath(os.path.join(outdir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_model_keep)

            sess.run(tf.global_variables_initializer())

            def viterbi_decoder(logits, seq_len_list, transition_params):
                label_list = []
                for logit, seq_len in zip(logits, seq_len_list):
                    viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
                    label_list.append(viterbi_seq)
                return label_list

            def train_step(x_batch, y_batch, sequence_lengths):
                feed_dict = {
                    bilstm_crf.input_x: x_batch,
                    bilstm_crf.input_y: y_batch,
                    bilstm_crf.sequence_length: sequence_lengths,
                    bilstm_crf.dropout_keep_prob: config['dropout_keep_prob']
                }

                _, step, summaries, loss, transition_params, logits = sess.run(
                    [train_op, global_step, train_sumary_op, bilstm_crf.loss,
                     bilstm_crf.transition_params, bilstm_crf.logits],
                    feed_dict=feed_dict
                )

                label_list = viterbi_decoder(logits, sequence_lengths, transition_params)

                acc, recall, f1 = data_helper.measure(y_batch, label_list, sequence_lengths)

                time_str = datetime.datetime.now().isoformat()
                print("training: {}: step {}, loss {:g}, acc {:.2f} recall {:.2f} f1 {:.2f}".format
                      (time_str, step, loss, acc, recall, f1))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, sequence_lengths, writer=None):
                feed_dic = {
                    bilstm_crf.input_x: x_batch,
                    bilstm_crf.input_y: y_batch,
                    bilstm_crf.sequence_length: sequence_lengths,
                    bilstm_crf.dropout_keep_prob: 1.0
                }

                step, summaries, loss, transition_params, logits = sess.run(
                    [global_step, dev_summary_op, bilstm_crf.loss, bilstm_crf.transition_params, bilstm_crf.logits],
                    feed_dict=feed_dic
                )

                label_list = viterbi_decoder(logits, sequence_lengths, transition_params)

                acc, recall, f1 = data_helper.measure(y_batch, label_list, sequence_lengths)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, f1 {:.2f}".format(time_str, step, loss, f1))
                if writer:
                    writer.add_summary(summaries, step)

            # generate batches
            batches = data_helper.generate_batchs(x_train, y_train, sequences_length_train, config)
            for batch in batches:
                x_batch, y_batch, sequence_length_batch = zip(*batch)
                train_step(x_batch, y_batch, sequence_length_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % config['evaluate_every'] == 0:
                    print('Evaluation:')
                    dev_step(x_dev, y_dev, sequence_length_dev, writer=dev_summary_writer)

                if current_step % config['checkpoint_every'] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('save model checkpoint to {}'.format(path))


if __name__ == '__main__':
    train(config)













