import tensorflow as tf
import numpy as np
import time
from Seq2SeqGraph import Seq2SeqGraph
from tensorflow.python.client import timeline

tf.set_random_seed(1)


class Seq2Seq:

    def __init__(self, params_common, params_model):
        self.params_common = params_common
        self.params_model = params_model

        with tf.variable_scope("seq2seq", reuse=tf.AUTO_REUSE):
            # *************** TRAIN GRAPH ***************
            print("="*60)
            print("# Building Train Graph")
            print("="*60)
            self.graph_train = Seq2SeqGraph(params_common, params_model, is_training=True)
            self.train_op, self.loss_op = self.get_train_op(self.graph_train.training_logits,
                                                            self.graph_train.target_output,
                                                            self.graph_train.target_sequence_length)

            # *************** INFER GRAPH ***************
            print("=" * 60)
            print("# Building Infer Graph")
            print("=" * 60)
            self.graph_infer = Seq2SeqGraph(params_common, params_model, is_training=False)

    def get_train_op(self, training_logits, target_output, target_sequence_length):
        # ------ BACKWARD -------
        # training phase, decode sequence length = true target seq length
        max_target_len = tf.reduce_max(target_sequence_length)
        masks = tf.sequence_mask(target_sequence_length, max_target_len, dtype=tf.float32, name="masks")
        with tf.name_scope("optimization"):
            loss_op = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                target_output,  # without <s>
                masks
            )

            optimizer = tf.train.AdamOptimizer(self.params_model["lr"])

            # Gradient Clipping 梯度裁剪
            gradients = optimizer.compute_gradients(loss_op)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gradients)
        return train_op, loss_op

    def train(self, sess, train_iter, valid_iter, sample_writer,
              options=None, run_metadata=None, train_timeline_fname=None):
        """

        :param sess:
        :param train_iter:
        :param valid_iter:
        :param params_common:
        :param sample_writer:
        :param options:
        :param run_metadata:
        :param train_timeline_fname:
        :return:
        """
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i_epoch in range(self.params_common["epochs"]):
            train_iter.reset()
            i_batch = 0
            train_time = 0
            while train_iter.has_next(self.params_common["batch_size"]):
                i_batch += 1
                start_train = time.clock()
                train_source_batch, train_target_batch,\
                train_source_lengths, train_target_lengths = train_iter.next_batch(self.params_common["batch_size"])

                # should run train_op to train, but only fetch cost
                # train phase的logit与input长度一定相同，才能计算loss

                _, train_batch_loss = sess.run([self.train_op, self.loss_op],
                                               options=options,
                                               run_metadata=run_metadata,
                                               feed_dict={self.graph_train.source_input: train_source_batch,
                                                          self.graph_train.source_sequence_length: train_source_lengths,
                                                          self.graph_train.target: train_target_batch,
                                                          self.graph_train.target_sequence_length: train_target_lengths})

                train_time += time.clock() - start_train

                # show progress
                if i_batch % self.params_common["valid_step"] == 0:
                    valid_iter.reset()
                    # ---- VALID ----
                    avg_precision = 0
                    num_valid_batch = 0
                    avg_valid_time = 0
                    avg_write_time = 0
                    with open(sample_writer.valid_infer_filename("s2satt", i_epoch), "w") as fsample:
                        while valid_iter.has_next(self.params_common["batch_size"]):
                            start_valid = time.clock()
                            num_valid_batch += 1
                            valid_source_batch, valid_target_batch,\
                            valid_source_lengths, valid_target_lengths = valid_iter.next_batch(self.params_common["batch_size"])

                            # inference的结果长度不一定与input一致！
                            infer_batch_logits, valid_target_output, infer_sequence_lengths = sess.run(
                                [self.graph_infer.inference_sample_id, self.graph_infer.target_output, self.graph_infer.infer_sequence_lengths],
                                options=options,
                                run_metadata=run_metadata,
                                feed_dict={self.graph_infer.source_input: valid_source_batch,
                                           self.graph_infer.source_sequence_length: valid_source_lengths,
                                           self.graph_infer.target: valid_target_batch,
                                           self.graph_infer.target_sequence_length: valid_target_lengths}
                            )
                            avg_valid_time += time.clock() - start_valid

                            # will rewrite at each valid step, finally keep one file for each epoch
                            start_write = time.clock()
                            sample_writer.write2file_inference_results(fsample, valid_source_batch, infer_batch_logits)
                            avg_write_time += time.clock() - start_write

                            # write out samples
                            if num_valid_batch % self.params_common["display_sample_per_n_batch"] == 0:
                                sample_writer.show_inference_samples(valid_source_batch, valid_target_output,
                                                                     infer_batch_logits, self.params_common["n_samples2write"])

                            # valid precision
                            valid_precision = self.__get_precision(sess, valid_target_output, infer_batch_logits,
                                                                   infer_sequence_lengths, self.params_common)
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
                    saver.save(sess, self.params_common["model_dir"] + "/"
                               + self.params_common["model_base"] + "-" + str(i_epoch) + "_" + str(i_batch),
                               )

        if run_metadata and options:
            start_timeline = time.clock()
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(train_timeline_fname, 'w') as f:
                f.write(chrome_trace)
            print("\t timeline %f s" % (time.clock() - start_timeline))

    def infer(self, sess, sequence, options=None, run_metadata=None, timeline_fname=None):
        """

        :param sequence: list of int
        :param params_common:
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(self.params_common["model_dir"]))
        # 若需要返回的结果不依赖于某个输入，feed_dict可以不给
        output_in_id = sess.run(self.graph_infer.inference_sample_id,
                                options=options,
                                run_metadata=run_metadata,
                                feed_dict={self.graph_infer.source_input: [sequence],
                                           self.graph_infer.source_sequence_length: [len(sequence)],
                                           self.graph_infer.target_sequence_length: [10]})[0]
        if run_metadata and options:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(timeline_fname + '.json', 'w') as f:
                f.write(chrome_trace)
        return output_in_id

    def __get_precision(self, sess, true_batch, pred_batch_logits, pred_seq_len, params_common):
        true_batch_maxlen = true_batch.shape[1]
        pred_batch_maxlen = pred_batch_logits.shape[1]
        max_seq_len = max(true_batch_maxlen, pred_batch_maxlen)
        if max_seq_len - true_batch_maxlen:
            # axis 0: (before, after), axis 1: (before, after)
            true_batch = np.pad(true_batch,
                                ((0, 0), (0, max_seq_len - true_batch.shape[1])),
                                "constant",
                                constant_values=((params_common["pad_id"], params_common["pad_id"]),
                                                 (params_common["pad_id"], params_common["pad_id"]))
                                )
        if max_seq_len - pred_batch_maxlen:
            pred_batch_logits = np.pad(pred_batch_logits,
                                       ((0,0), (0, max_seq_len - pred_batch_logits.shape[1])),
                                       "constant")

        # pred_seq_len = self.__length(pred_batch_logits, params["pad_id"])

        mask = tf.cast(tf.sequence_mask(pred_seq_len, max_seq_len), tf.int32)
        equals = tf.cast(tf.equal(pred_batch_logits, true_batch), tf.int32)
        masked_eq = tf.multiply(equals, mask)
        true_pos = tf.reduce_sum(masked_eq)
        all = tf.reduce_sum(pred_seq_len)
        return sess.run(true_pos / all)
        # accuracy (including pad)
        # return np.mean(np.equal(true_batch, pred_batch_logits))
