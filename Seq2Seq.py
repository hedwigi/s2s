import tensorflow as tf
import numpy as np
import time
from util.Monitor import Monitor
from tensorflow.python.client import timeline

tf.set_random_seed(1)


class Seq2Seq(object):

    def __init__(self, params):
        """

        :param params:
        """
        # *************** PLACEHOLDER & INPUT ***************
        # [batch_size, sequence_len]
        self.source_input = tf.placeholder(tf.int32, [None, None], name="source_input")
        self.target = tf.placeholder(tf.int32, [None, None], name="target")

        # if reverse_target, length doesn't include <S> at the end
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
        max_target_len = tf.reduce_max(self.target_sequence_length)

        if params["reverse_target"]:
            # target input: <EOS> 3 2 1 <S> <PAD>
            target_input = self.target
            # target output: 3 2 1 <S> <PAD> <PAD>
            first_slices = tf.strided_slice(target_input, [0,1], [params["batch_size"], max_target_len], [1, 1])
            self.target_output = tf.concat([first_slices, tf.fill([params["batch_size"], 1], params["start_id"])], 1)
        else:
            # target output: 1 2 3 <EOS>
            self.target_output = self.target
            # target input: <S> 1 2 3
            after_slice = tf.strided_slice(self.target_output, [0, 0], [params["batch_size"], -1], [1, 1])
            target_input = tf.concat([tf.fill([params["batch_size"], 1], params["start_id"]), after_slice], 1)

        # *************** GRAPH ****************
        # ------ RNN Encoder ------
        # TODO change to independent embedding
        embed = tf.contrib.layers.embed_sequence(self.source_input, vocab_size=params["source_vocab_size"],
                                                 embed_dim=params["encoding_embedding_size"])
        # (uni-rnn)
        # list of separated rnn cells
        # rnn_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params["rnn_size"]), params["keep_prob"])

        # stack n layers together
        # stacked_cells = tf.contrib.rnn.MultiRNNCell(l_dropped_out_rnn_cell)
        # unroll rnn_cell instance, output是通过主动提供输入tok得到的
        # _, encoder_state = tf.nn.dynamic_rnn(rnn_cell, embed, dtype=tf.float32)

        # (bi-rnn)
        input = embed
        encoder_states = []
        for _ in range(params["num_layers"]):
            with tf.variable_scope(None, default_name="stacked_bilstm"):
                fw_rnn_cell = tf.contrib.rnn.LSTMCell(params["rnn_size"] / 2)
                fw_dropped_out_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, params["keep_prob"])

                bw_rnn_cell = tf.contrib.rnn.LSTMCell(params["rnn_size"] / 2)
                bw_dropped_out_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, params["keep_prob"])

                # state = (fw_state, bw_state)
                # fw_state = (c, h)
                outputs, state = tf.nn.bidirectional_dynamic_rnn(
                    fw_dropped_out_rnn_cell, bw_dropped_out_rnn_cell, input, dtype=tf.float32
                )

                # update
                input = tf.concat(outputs, 2)
                states = tf.concat([state[0], state[1]], axis=2)
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(states[0], states[1])
                encoder_states.append(encoder_state)
        encoder_states = tuple(encoder_states)

        # ------ RNN Decoder -------
        dec_embeddings = tf.Variable(tf.random_uniform([params["target_vocab_size"], params["decoding_embedding_size"]]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, target_input)

        l_dec_rnn_cell = [tf.contrib.rnn.LSTMCell(params["rnn_size"]) for i in range(params["num_layers"])]
        dec_stacked_cells = tf.contrib.rnn.MultiRNNCell(l_dec_rnn_cell)
        # reuse: shared rnn cells
        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
            # --- Train phase ---
            dec_train_cells = tf.contrib.rnn.DropoutWrapper(dec_stacked_cells, output_keep_prob=params["keep_prob"])
            output_layer = tf.layers.Dense(params["target_vocab_size"])

            # dynamic_rnn只能使用提供的input得到output，（helper + decoder + dynamic_decode）可以自定义得到output的方式
            # 由helper决定decoder的input。此处dec_embed是true label的输入tok
            helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, self.target_sequence_length)

            # 核心decoder，使用helper的input和rnn_cell，以及输出层，返回单次的RNN output
            decoder_train = tf.contrib.seq2seq.BasicDecoder(dec_train_cells, helper, encoder_states, output_layer)

            # 使用核心decoder，提供用来unroll的大循环
            self.decoder_train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=max_target_len)

            # --- Infer phase ---
            # TODO: another Dropout????
            dec_infer_cells = tf.contrib.rnn.DropoutWrapper(dec_stacked_cells, output_keep_prob=params["keep_prob"])
            infer_start = params["end_id"] if params["reverse_target"] else params["start_id"]
            infer_end = params["start_id"] if params["reverse_target"] else params["end_id"]
            gd_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                 tf.fill([params["batch_size"]], infer_start),
                                                                 infer_end)
            decoder_infer = tf.contrib.seq2seq.BasicDecoder(dec_infer_cells, gd_helper, encoder_states, output_layer)
            self.decoder_infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=max_target_len)

        # ------ TRAIN -------
        # TODO: with same value, why need identity?
        training_logits = tf.identity(self.decoder_train_outputs.rnn_output, name="logits")
        self.inference_sample_id = tf.identity(self.decoder_infer_outputs.sample_id, name="predictions")

        masks = tf.sequence_mask(self.target_sequence_length, max_target_len, dtype=tf.float32, name="masks")
        with tf.name_scope("optimization"):
            self.cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                self.target_output,  # without <s>
                masks
            )

            optimizer = tf.train.AdamOptimizer(params["lr"])

            # Gradient Clipping 梯度裁剪
            gradients = optimizer.compute_gradients(self.cost)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(clipped_gradients)

        vars = tf.trainable_variables()
        Monitor.print_params(vars)

    def train(self, sess, train_dataset, valid_dataset, params, sample_writer, options, run_metadata):
        """

        :param sess:
        :param train_dataset:
        :param valid_dataset:
        :param params:
        :param sample_writer:
        :param options:
        :param run_metadata:
        :return:
        """
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i_epoch in range(params["epochs"]):
            train_dataset.reset()
            i_batch = 0
            train_time = 0
            while train_dataset.has_next(params["batch_size"]):
                i_batch += 1
                start_train = time.clock()
                train_source_batch, train_target_batch,\
                _, train_target_lengths = train_dataset.next_batch(params["batch_size"])
                # should run train_op to train, but only fetch cost
                # train phase的logit与input长度一定相同，才能计算loss
                _, train_batch_loss = sess.run([self.train_op, self.cost],
                                               options=options,
                                               run_metadata=run_metadata,
                                   feed_dict={self.source_input: train_source_batch,
                                            self.target: train_target_batch,
                                            self.target_sequence_length: train_target_lengths})
                train_time += time.clock() - start_train

                # show progress
                if i_batch % params["valid_step"] == 0:
                    valid_dataset.reset()
                    # ---- VALID ----
                    avg_acc = 0
                    num_valid_batch = 0
                    start_valid = 0
                    while valid_dataset.has_next(params["batch_size"]):
                        num_valid_batch += 1
                        valid_source_batch, valid_target_batch,\
                        valid_source_lengths, valid_target_lengths = valid_dataset.next_batch(params["batch_size"])

                        # inference的结果长度不一定与input一致！
                        valid_batch_logits, valid_target_output = sess.run(
                            [self.inference_sample_id, self.target_output],
                            options=options,
                            run_metadata=run_metadata,
                            feed_dict={self.source_input: valid_source_batch,
                                       self.target: valid_target_batch,
                                       self.target_sequence_length: valid_target_lengths}
                        )

                        # write out samples
                        if num_valid_batch % params["display_sample_per_n_batch"] == 0:
                            sample_writer.write_inference_samples(valid_source_batch, valid_batch_logits, params["n_samples2write"])

                        valid_acc = self.__get_accuracy(valid_target_output, valid_batch_logits, params)
                        avg_acc += valid_acc
                    avg_acc /= num_valid_batch
                    valid_time = time.clock() - start_valid

                    print("Epoch %d, Batch %d - Valid acc: %f, Train batch loss: %f, "
                          "AVG train time per batch: %f s, AVG valid time per batch %f s"
                          % (i_epoch, i_batch, avg_acc, train_batch_loss,
                             train_time / i_batch,
                             valid_time / num_valid_batch
                             ))

                    # 在每次print的时候save，使得print的结果与保存的model相对应
                    saver.save(sess, params["model_dir"] + "/"
                               + params["model_base"] + "-" + str(i_epoch) + "_" + str(i_batch),
                               )

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)

    def infer(self, sess, sequence, params):
        """

        :param sequence: list of int
        :param params:
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(params["model_dir"]))
        # 若需要返回的结果不依赖于某个输入，feed_dict可以不给
        output_in_id = sess.run(self.inference_sample_id, feed_dict={self.source_input: [sequence]})[0]
        return output_in_id

    def __get_accuracy(self, true_batch, pred_batch_logits, params):
        """

        :param true_batch:
        :param pred_batch_logits:
        :return:
        """
        true_batch_seqlen = true_batch.shape[1]
        pred_batch_seqlen = pred_batch_logits.shape[1]
        max_seq_len = max(true_batch_seqlen, pred_batch_seqlen)
        if max_seq_len - true_batch_seqlen:
            # axis 0: (before, after), axis 1: (before, after)
            true_batch = np.pad(true_batch,
                                ((0, 0), (0, max_seq_len - true_batch.shape[1])),
                                "constant",
                                constant_values=((params["pad_id"], params["pad_id"]),
                                                 (params["pad_id"], params["pad_id"]))
                                )
        if max_seq_len - pred_batch_seqlen:
            pred_batch_logits = np.pad(pred_batch_logits,
                                       ((0,0), (0, max_seq_len - pred_batch_logits.shape[1])),
                                       "constant")
        # reduce mean
        return np.mean(np.equal(true_batch, pred_batch_logits))
