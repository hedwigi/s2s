from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import numpy as np

from TransformerKyuGraph import TransformerKyuGraph


class TransformerKyu:
    def __init__(self, params):
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            self.graph_train = TransformerKyuGraph(params, is_training=True)
            self.graph_infer = TransformerKyuGraph(params, is_training=False)

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
                _, train_batch_loss = sess.run([self.graph_train.train_op, self.graph_train.mean_loss],
                                               options=options,
                                               run_metadata=run_metadata,
                                               feed_dict={self.graph_train.source_input: train_source_batch,
                                                          self.graph_train.source_sequence_length: train_source_lengths,
                                                          self.graph_train.target: train_target_batch,
                                                          self.graph_train.target_sequence_length: train_target_lengths})
                train_time += time.clock() - start_train

                # show progress
                if i_batch % params["valid_step"] == 0:
                    valid_dataset.reset()
                    # ---- VALID ----
                    avg_precision = 0
                    num_valid_batch = 0
                    avg_valid_time = 0
                    avg_write_time = 0
                    with open(sample_writer.valid_infer_filename("trfm", i_epoch), "w") as fsample:
                        while valid_dataset.has_next(params["batch_size"]):
                            start_valid = time.clock()
                            num_valid_batch += 1
                            valid_source_batch, valid_target_batch, \
                            valid_source_lengths, valid_target_lengths = valid_dataset.next_batch(params["batch_size"])

                            # Autoregressive inference
                            infer_result = np.zeros((params["batch_size"], params["maxlen"]), np.int32)
                            for j in range(params["maxlen"]):
                                _preds = sess.run(self.graph_infer.preds,
                                                  feed_dict={self.graph_infer.source_input: valid_source_batch,
                                                               self.graph_infer.target: infer_result})
                                infer_result[:, j] = _preds[:, j]
                            infer_sequence_lengths = np.sum(np.not_equal(infer_result, params["pad_id"]),
                                                            axis=-1, dtype=np.float32)

                            # inference的结果长度不一定与input一致！
                            # infer_batch_logits, valid_target_output, infer_sequence_lengths = sess.run(
                            #     [self.preds, self.target, self.infer_sequence_lengths],
                            #     options=options,
                            #     run_metadata=run_metadata,
                            #     feed_dict={self.source_input: valid_source_batch,
                            #                self.source_sequence_length: valid_source_lengths,
                            #                self.target: valid_target_batch,
                            #                self.target_sequence_length: valid_target_lengths}
                            # )
                            avg_valid_time += time.clock() - start_valid

                            # will rewrite at each valid step, finally keep one file for each epoch
                            start_write = time.clock()
                            sample_writer.write2file_inference_results(fsample, valid_source_batch, infer_result)
                            avg_write_time += time.clock() - start_write

                            # write out samples
                            if num_valid_batch % params["display_sample_per_n_batch"] == 0:
                                sample_writer.show_inference_samples(valid_source_batch, valid_target_batch,
                                                                     infer_result,
                                                                     params["n_samples2write"])

                            # valid precision
                            valid_precision = self.__get_precision(sess, valid_target_batch, infer_result,
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


