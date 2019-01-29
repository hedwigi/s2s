"""
Created on 2019/1/28

@author: wangyuqian
"""
import time
import numpy as np
import tensorflow as tf
from TransformerGraph import TransformerGraph
from tensorflow.python.client import timeline
from util.Monitor import Monitor


class Transformer:
    def __init__(self, params_common, params_model):
        self.params_common = params_common
        self.params_model = params_model

        # *************** PLACEHOLDER & INPUT ***************
        # [batch_size, sequence_len]
        self.source_input = tf.placeholder(tf.int32, [None, None], name="source_input")
        self.target = tf.placeholder(tf.int32, [None, None], name="target")

        self.source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
        # if reverse_target, length doesn't include <S> at the end
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")

        with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
            # *************** TRAIN GRAPH ***************
            print("=" * 60)
            print("# Building Train Graph")
            print("=" * 60)
            self.graph_train = TransformerGraph(params_common, params_model, train=True)
            self.training_logits = self.graph_train(self.source_input, self.target)
            # TODO: if reverse target, target output != self.target
            self.train_op, self.loss_op = self.get_train_op(self.training_logits, self.target,
                                                            self.target_sequence_length)
            vars = tf.trainable_variables()
            Monitor.print_params(vars)

            # *************** INFER GRAPH ***************
            print("=" * 60)
            print("# Building Infer Graph")
            print("=" * 60)
            self.graph_infer = TransformerGraph(params_common, params_model, train=False)
            infer_outputs = self.graph_infer(self.source_input)
            self.inference_sample_id = infer_outputs["outputs"]
            ones = tf.ones_like(self.inference_sample_id)
            isinferred = tf.where(tf.not_equal(self.inference_sample_id, params_common["pad_id"]),
                                  ones, 1-ones)
            self.infer_sequence_lengths = tf.reduce_sum(isinferred, axis=-1)
            vars = tf.trainable_variables()
            Monitor.print_params(vars)

    def get_train_op(self, training_logits, target_output, target_sequence_length):
        # ------ BACKWARD -------
        with tf.name_scope("optimization"):
            # Calculate model loss.
            # xentropy contains the cross entropy loss of every nonpadding token in the
            # targets.
            xentropy, weights = self.__padded_cross_entropy_loss(
                training_logits, target_output, self.params_model["label_smoothing"], self.params_model["vocab_size"])
            loss_op = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
            train_op = self.__get_train_op_and_metrics(loss_op, self.params_model)
        return train_op, loss_op

    def train(self, sess, train_iter, valid_iter, sample_writer,
              options, run_metadata, train_timeline_fname):

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i_epoch in range(self.params_common["epochs"]):
            train_iter.reset()
            i_batch = 0
            train_time = 0
            while train_iter.has_next(self.params_common["batch_size"]):
                i_batch += 1
                start_train = time.clock()
                train_source_batch, train_target_batch, \
                train_source_lengths, train_target_lengths = train_iter.next_batch(self.params_common["batch_size"])
                # should run train_op to train, but only fetch cost
                # train phase的logit与input长度一定相同，才能计算loss

                # train loss
                _, train_batch_loss = sess.run([self.train_op, self.loss_op],
                                               feed_dict={self.source_input: train_source_batch,
                                                          self.source_sequence_length: train_source_lengths,
                                                          self.target: train_target_batch,
                                                          self.target_sequence_length: train_target_lengths})

                train_time += time.clock() - start_train

                # show progress
                if i_batch % self.params_common["valid_step"] == 0:
                    valid_iter.reset()
                    # ---- VALID ----
                    avg_precision = 0
                    num_valid_batch = 0
                    avg_valid_time = 0
                    avg_write_time = 0
                    with open(sample_writer.valid_infer_filename(self.params_common["results_base"], i_epoch), "w") as fsample:
                        while valid_iter.has_next(self.params_common["batch_size"]):
                            start_valid = time.clock()
                            num_valid_batch += 1
                            valid_source_batch, valid_target_batch, \
                            valid_source_lengths, valid_target_lengths = valid_iter.next_batch(self.params_common["batch_size"])

                            # inference的结果长度不一定与input一致！
                            infer_batch_logits, infer_sequence_lengths = sess.run(
                                [self.inference_sample_id, self.infer_sequence_lengths],
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
                            if num_valid_batch % self.params_common["display_sample_per_n_batch"] == 0:
                                sample_writer.show_inference_samples(valid_source_batch,
                                                                     valid_target_batch,
                                                                     infer_batch_logits,
                                                                     self.params_common["n_samples2write"])

                            # valid precision
                            valid_precision = self.__get_precision(sess, valid_target_batch, infer_batch_logits,
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

    def infer(self, sess, question_in_id, options, run_metadata, valid_timeline_fname):
        pass

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
        true_pos = tf.reduce_sum(masked_eq)
        all = tf.reduce_sum(pred_seq_len)
        return sess.run(true_pos / all)
        # accuracy (including pad)
        # return np.mean(np.equal(true_batch, pred_batch_logits))

    def __padded_cross_entropy_loss(self, logits, labels, smoothing, vocab_size):
        """Calculate cross entropy loss while ignoring padding.

        Args:
          logits: Tensor of size [batch_size, length_logits, vocab_size]
          labels: Tensor of size [batch_size, length_labels]
          smoothing: Label smoothing constant, used to determine the on and off values
          vocab_size: int size of the vocabulary
        Returns:
          Returns the cross entropy loss and weight tensors: float32 tensors with
            shape [batch_size, max(length_logits, length_labels)]
        """
        with tf.name_scope("loss", [logits, labels]):
            logits, labels = self._pad_tensors_to_same_length(logits, labels)

            # Calculate smoothing cross entropy
            with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
                confidence = 1.0 - smoothing
                low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
                soft_targets = tf.one_hot(
                    tf.cast(labels, tf.int32),
                    depth=vocab_size,
                    on_value=confidence,
                    off_value=low_confidence)
                xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=soft_targets)

                # Calculate the best (lowest) possible value of cross entropy, and
                # subtract from the cross entropy loss.
                normalizing_constant = -(
                        confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                        low_confidence * tf.log(low_confidence + 1e-20))
                xentropy -= normalizing_constant

            weights = tf.to_float(tf.not_equal(labels, 0))
            return xentropy * weights, weights

    def _pad_tensors_to_same_length(self, x, y):
        """Pad x and y so that the results have the same length (second dimension)."""
        with tf.name_scope("pad_to_same_length"):
            x_length = tf.shape(x)[1]
            y_length = tf.shape(y)[1]

            max_length = tf.maximum(x_length, y_length)

            x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
            y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
            return x, y

    def __get_train_op_and_metrics(self, loss, params):
        """Generate training op and metrics to save in TensorBoard."""
        with tf.variable_scope("get_train_op"):
            learning_rate = self.__get_learning_rate(
                learning_rate=params["learning_rate"],
                hidden_size=params["hidden_size"],
                learning_rate_warmup_steps=params["learning_rate_warmup_steps"])

            # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
            # than the TF core Adam optimizer.
            optimizer = tf.contrib.opt.LazyAdamOptimizer(
                learning_rate,
                beta1=params["optimizer_adam_beta1"],
                beta2=params["optimizer_adam_beta2"],
                epsilon=params["optimizer_adam_epsilon"])

            # Calculate and apply gradients using LazyAdamOptimizer.
            global_step = tf.train.get_global_step()
            tvars = tf.trainable_variables()
            gradients = optimizer.compute_gradients(
                loss, tvars, colocate_gradients_with_ops=True)
            minimize_op = optimizer.apply_gradients(
                gradients, global_step=global_step, name="train")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)

            return train_op

    def __get_learning_rate(self, learning_rate, hidden_size, learning_rate_warmup_steps):
        """Calculate learning rate with linear warmup and rsqrt decay."""
        with tf.name_scope("learning_rate"):
            warmup_steps = tf.to_float(learning_rate_warmup_steps)
            step = tf.to_float(tf.train.get_or_create_global_step())

            learning_rate *= (hidden_size ** -0.5)
            # Apply linear warmup
            learning_rate *= tf.minimum(1.0, step / warmup_steps)
            # Apply rsqrt decay
            learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

            # Create a named tensor that will be logged using the logging hook.
            # The full name includes variable and names scope. In this case, the name
            # is model/get_train_op/learning_rate/learning_rate
            tf.identity(learning_rate, "learning_rate")

            return learning_rate
