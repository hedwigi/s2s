from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import numpy as np

from util.TransformerUtil import *


class TransformerKyu:
    def __init__(self, params, is_training=True):
        self.source_input = tf.placeholder(tf.int32, shape=(None, None), name="source_input")
        self.target = tf.placeholder(tf.int32, shape=(None, None), name="target")
        self.source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
        # if reverse_target, length doesn't include <S> at the end
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")

        # 获取max target len
        max_source_len = tf.reduce_max(self.source_sequence_length)

        # 获取可变的batch_size
        batch_size = tf.shape(self.source_input)[0]

        # define decoder inputs
        self.decoder_inputs = tf.concat((tf.ones_like(self.target[:, :1]) * 2, self.target[:, :-1]), -1)  # 2:<S>

        # Encoder
        with tf.variable_scope("encoder"):
            # Embedding
            self.enc = embedding(self.source_input,
                                 vocab_size=params["source_vocab_size"],
                                 num_units=params["hidden_units"],
                                 scale=True,
                                 scope="enc_embed")

            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)

            # Positional Encoding
            if params["sinusoid"]:
                self.enc += positional_encoding(self.source_input,
                                                num_units=params["hidden_units"],
                                                zero_pad=False,
                                                scale=False,
                                                scope="enc_pe")
            else:
                self.enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.source_input)[1]), 0), [tf.shape(self.source_input)[0], 1]),
                    vocab_size=params["maxlen"] * params["max_context_size"],
                    num_units=params["hidden_units"],
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")

            self.enc *= key_masks

            # Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=params["dropout_rate"],
                                         training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(params["num_blocks"]):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=params["hidden_units"],
                                                   num_heads=params["num_heads"],
                                                   dropout_rate=params["dropout_rate"],
                                                   is_training=is_training,
                                                   causality=False)

                    ## Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * params["hidden_units"], params["hidden_units"]])

        # Decoder
        with tf.variable_scope("decoder"):
            # Embedding
            self.dec = embedding(self.decoder_inputs,
                                 vocab_size=params["target_vocab_size"],
                                 num_units=params["hidden_units"],
                                 scale=True,
                                 scope="dec_embed")

            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)), -1)

            # Positional Encoding
            if params["sinusoid"]:
                self.dec += positional_encoding(self.decoder_inputs,
                                                num_units=params["hidden_units"],
                                                zero_pad=False,
                                                scale=False,
                                                scope="dec_pe")
            else:
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                                              [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=params["maxlen"],
                                      num_units=params["hidden_units"],
                                      zero_pad=False,
                                      scale=False,
                                      scope="dec_pe")
            self.dec *= key_masks

            # Dropout
            self.dec = tf.layers.dropout(self.dec,
                                         rate=params["dropout_rate"],
                                         training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(params["num_blocks"]):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.dec,
                                                   num_units=params["hidden_units"],
                                                   num_heads=params["num_heads"],
                                                   dropout_rate=params["dropout_rate"],
                                                   is_training=is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=params["hidden_units"],
                                                   num_heads=params["num_heads"],
                                                   dropout_rate=params["dropout_rate"],
                                                   is_training=is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")

                    # Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4 * params["hidden_units"], params["hidden_units"]])

        # Final linear projection
        self.logits = tf.layers.dense(self.dec, params["target_vocab_size"])
        # beam size = 1
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.target, 0))  # 0: PAD
        self.infer_sequence_lengths = tf.reduce_sum(self.istarget, axis=-1)
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.target)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.target, depth=params["target_vocab_size"]))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"], beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()

    def train(self, sess, train_dataset, valid_dataset, params, sample_writer,
              options=None, run_metadata=None, train_timeline_fname=None):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i_epoch in range(params["epochs"]):
            train_dataset.reset()
            i_batch = 0
            train_time = 0
            while train_dataset.has_next(params["batch_size"]):
                i_batch += 1
                start_train = time.clock()
                train_source_batch, train_target_batch, \
                train_source_lengths, train_target_lengths = train_dataset.next_batch(params["batch_size"])
                # should run train_op to train, but only fetch cost
                # train phase的logit与input长度一定相同，才能计算loss
                _, train_batch_loss = sess.run([self.train_op, self.mean_loss],
                                               options=options,
                                               run_metadata=run_metadata,
                                               feed_dict={self.source_input: train_source_batch,
                                                          self.source_sequence_length: train_source_lengths,
                                                          self.target: train_target_batch,
                                                          self.target_sequence_length: train_target_lengths})
                train_time += time.clock() - start_train

                # show progress
                if i_batch % params["valid_step"] == 0:
                    valid_dataset.reset()
                    # ---- VALID ----
                    avg_precision = 0
                    num_valid_batch = 0
                    avg_valid_time = 0
                    avg_write_time = 0
                    with open(sample_writer.valid_infer_filename(i_epoch), "w") as fsample:
                        while valid_dataset.has_next(params["batch_size"]):
                            start_valid = time.clock()
                            num_valid_batch += 1
                            valid_source_batch, valid_target_batch, \
                            valid_source_lengths, valid_target_lengths = valid_dataset.next_batch(params["batch_size"])

                            # inference的结果长度不一定与input一致！
                            infer_batch_logits, valid_target_output, infer_sequence_lengths = sess.run(
                                [self.preds, self.target, self.infer_sequence_lengths],
                                options=options,
                                run_metadata=run_metadata,
                                feed_dict={self.source_input: valid_source_batch,
                                           self.source_sequence_length: valid_source_lengths,
                                           self.target: valid_target_batch,
                                           self.target_sequence_length: valid_target_lengths}
                            )
                            avg_valid_time += time.clock() - start_valid

                            # will rewrite at each valid step, finally keep one file for each epoch
                            start_write = time.clock()
                            sample_writer.write2file_inference_results(fsample, valid_source_batch, infer_batch_logits)
                            avg_write_time += time.clock() - start_write

                            # write out samples
                            if num_valid_batch % params["display_sample_per_n_batch"] == 0:
                                sample_writer.show_inference_samples(valid_source_batch, infer_batch_logits,
                                                                     params["n_samples2write"])

                            # valid precision
                            valid_precision = self.__get_precision(sess, valid_target_output, infer_batch_logits,
                                                                   infer_sequence_lengths, params)
                            avg_precision += valid_precision
                        avg_precision /= num_valid_batch
                        avg_valid_time /= num_valid_batch
                        avg_write_time /= num_valid_batch

                    print("Epoch %d, Batch %d - Valid precision: %f, Train batch loss: %f, "
                          "AVG train time per batch: %f s, AVG valid time per batch %f s, "
                          "AVG write time per batch: %f s on %d batches"
                          % (i_epoch, i_batch, avg_precision, train_batch_loss,
                             train_time / i_batch,
                             avg_valid_time,
                             avg_write_time,
                             num_valid_batch))

                    # 在每次print的时候save，使得print的结果与保存的model相对应
                    saver.save(sess, params["model_dir"] + "/"
                               + params["model_base"] + "-" + str(i_epoch) + "_" + str(i_batch),
                               )

        if run_metadata and options:
            start_timeline = time.clock()
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(train_timeline_fname, 'w') as f:
                f.write(chrome_trace)
            print("\t timeline %f s" % (time.clock() - start_timeline))

    def __get_precision(self, sess, true_batch, pred_batch_logits, pred_seq_len, params):
        true_batch_maxlen = true_batch.shape[1]
        pred_batch_maxlen = pred_batch_logits.shape[1]
        max_seq_len = max(true_batch_maxlen, pred_batch_maxlen)
        if max_seq_len - true_batch_maxlen:
            # axis 0: (before, after), axis 1: (before, after)
            true_batch = np.pad(true_batch,
                                ((0, 0), (0, max_seq_len - true_batch.shape[1])),
                                "constant",
                                constant_values=((params["pad_id"], params["pad_id"]),
                                                 (params["pad_id"], params["pad_id"]))
                                )
        if max_seq_len - pred_batch_maxlen:
            pred_batch_logits = np.pad(pred_batch_logits,
                                       ((0,0), (0, max_seq_len - pred_batch_logits.shape[1])),
                                       "constant")

        # pred_seq_len = self.__length(pred_batch_logits, params["pad_id"])

        mask = tf.cast(tf.sequence_mask(pred_seq_len, max_seq_len), tf.int32)
        equals = tf.cast(tf.equal(pred_batch_logits, true_batch), tf.int32)
        masked_eq = tf.multiply(equals, mask)
        true_pos = tf.cast(tf.reduce_sum(masked_eq), tf.float32)
        all = tf.reduce_sum(pred_seq_len)
        return sess.run(true_pos / all)
        # accuracy (including pad)
        # return np.mean(np.equal(true_batch, pred_batch_logits))

# if __name__ == '__main__':
#     # Load vocabulary
#     source_vocab2idx, idx2source_vocab = load_source_vocab()
#     target_vocab2idx, idx2target_vocab = load_target_vocab()
#
#     # Params
#     train_mode = True
#     n_sample2show = hp.n_sample2show
#     n_show_step = hp.n_show_step
#
#     # Construct graph
#     graph = TransformerKyu(train_mode)
#     print("Graph loaded")
#
#     # Start session
#     print("Need to clear all checkpoint and models in logdir/ !!!!!")
#     sv = tf.train.Supervisor(graph=graph.graph,
#                              logdir=hp.logdir,
#                              save_model_secs=0)
#     with sv.managed_session() as sess:
#         for epoch in range(1, hp.num_epochs + 1):
#             if sv.should_stop(): break
#             for step in tqdm(range(graph.num_batch), total=graph.num_batch, ncols=70, leave=False, unit='b'):
#                 _, mean_loss, acc, preds, y, x = sess.run([graph.train_op, graph.mean_loss, graph.acc,
#                                                            graph.preds, graph.y, graph.x])
#
#                 if step % n_show_step == 0:
#                     # show samples
#                     for p in range(n_sample2show):
#                         xx = " ".join(idx2source_vocab[idx] for idx in x[p]).split("</S>")[0].strip()
#                         yy = " ".join(idx2target_vocab[idx] for idx in y[p]).split("</S>")[0].strip()
#                         pre = " ".join(idx2target_vocab[idx] for idx in preds[p]).split("</S>")[0].strip()
#                         print("\tSource: %s ### Target: %s ### Pred: %s" % (xx, yy, pre))
#
#                 print("Epoch %d, Step %d, Mean train loss %f, Train acc %f" % (epoch, step, mean_loss, acc))
#
#             gs = sess.run(graph.global_step)
#             sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
#
#     print("Done")


