import tensorflow as tf
import numpy as np

tf.set_random_seed(1)


class Seq2Seq(object):

    def __init__(self, params):
        """

        :param params:
        """
        # *************** PLACEHOLDER & INPUT ***************
        # [batch_size, sequence_len]
        self.source_input = tf.placeholder(tf.int32, [None, None], name="source_input")
        self.target_output = tf.placeholder(tf.int32, [None, None], name="target_output")

        # Todo strided_slice
        after_slice = tf.strided_slice(self.target_output, [0, 0], [params["batch_size"], -1], [1, 1])
        self.target_input = tf.concat([tf.fill([params["batch_size"], 1], params["start_id"]), after_slice], 1)

        # TODO why need target_ids len???
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
        max_target_len = tf.reduce_max(self.target_sequence_length)

        # *************** GRAPH ****************
        # ------ RNN Encoder ------
        # TODO change to independent embedding
        embed = tf.contrib.layers.embed_sequence(self.source_input, vocab_size=params["source_vocab_size"],
                                                 embed_dim=params["encoding_embedding_size"])
        # list of separated rnn cells
        l_rnn_cell = [tf.contrib.rnn.LSTMCell(params["rnn_size"]) for _ in range(params["num_layers"])]
        # dropout
        l_dropped_out_rnn_cell = [tf.contrib.rnn.DropoutWrapper(cell, params["keep_prob"]) for cell in l_rnn_cell]
        # stack n layers together
        stacked_cells = tf.contrib.rnn.MultiRNNCell(l_dropped_out_rnn_cell)
        # unroll rnn_cell instance, output是通过主动提供输入tok得到的
        _, encoder_state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)

        # ------ RNN Decoder -------
        dec_embeddings = tf.Variable(tf.random_uniform([params["target_vocab_size"], params["decoding_embedding_size"]]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, self.target_input)

        l_dec_rnn_cell = [tf.contrib.rnn.LSTMCell(params["rnn_size"]) for _ in range(params["num_layers"])]
        dec_stacked_cells = tf.contrib.rnn.MultiRNNCell(l_dec_rnn_cell)

        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
            # --- Train phase ---
            dec_train_cells = tf.contrib.rnn.DropoutWrapper(dec_stacked_cells, output_keep_prob=params["keep_prob"])
            output_layer = tf.layers.Dense(params["target_vocab_size"])

            # dynamic_rnn只能使用提供的input得到output，（helper + decoder + dynamic_decode）可以自定义得到output的方式
            # 由helper决定decoder的input。此处dec_embed是true label的输入tok
            helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, self.target_sequence_length)

            # 核心decoder，使用helper的input和rnn_cell，以及输出层，返回单次的RNN output
            decoder_train = tf.contrib.seq2seq.BasicDecoder(dec_train_cells, helper, encoder_state, output_layer)

            # 使用核心decoder，提供用来unroll的大循环
            self.decoder_train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=max_target_len)

            # --- Infer phase ---
            # TODO: another Dropout????
            dec_infer_cells = tf.contrib.rnn.DropoutWrapper(dec_stacked_cells, output_keep_prob=params["keep_prob"])
            gd_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                 tf.fill([params["batch_size"]], params["start_id"]),
                                                                 params["end_id"])
            decoder_infer = tf.contrib.seq2seq.BasicDecoder(dec_infer_cells, gd_helper, encoder_state, output_layer)
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

    def train(self, sess, train_dataset, valid_dataset, params, sample_writer):
        """

        :param sess:
        :param train_dataset:
        :param valid_dataset:
        :param params:
        :param sample_writer:
        :return:
        """
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i_epoch in range(params["epochs"]):
            train_dataset.reset()
            i_batch = 0
            while train_dataset.has_next(params["batch_size"]):
                i_batch += 1
                train_source_batch, train_target_batch,\
                _, train_target_lengths = train_dataset.next_batch(params["batch_size"])
                # should run train_op to train, but only fetch cost
                # train phase的logit与input长度一定相同，才能计算loss
                _, train_batch_loss = sess.run([self.train_op, self.cost],
                                   feed_dict={self.source_input: train_source_batch,
                                            self.target_output: train_target_batch,
                                            self.target_sequence_length: train_target_lengths})

                # show progress
                if i_batch % params["valid_step"] == 0:
                    valid_dataset.reset()
                    # ---- VALID ----
                    avg_acc = 0
                    num_valid_batch = 0
                    while valid_dataset.has_next(params["batch_size"]):
                        num_valid_batch += 1
                        valid_source_batch, valid_target_batch,\
                        valid_source_lengths, valid_target_lengths = valid_dataset.next_batch(params["batch_size"])

                        # inference的结果长度不一定与input一致！
                        valid_batch_logits = sess.run(
                            self.inference_sample_id,
                            feed_dict={self.source_input: valid_source_batch,
                                       self.target_output: valid_target_batch,
                                       self.target_sequence_length: valid_target_lengths}
                        )

                        # write out samples
                        if num_valid_batch % params["display_sample_per_n_batch"] == 0:
                            sample_writer.write_inference_samples(valid_source_batch, valid_batch_logits, params["n_samples2write"])

                        valid_acc = self.__get_accuracy(valid_target_batch, valid_batch_logits, params)
                        avg_acc += valid_acc
                    avg_acc /= num_valid_batch

                    print("Epoch %d, Batch %d - Valid acc: %f, Train batch loss: %f"
                          % (i_epoch, i_batch, avg_acc, train_batch_loss))

                    # 在每次print的时候save，使得print的结果与保存的model相对应
                    saver.save(sess, params["model_dir"] + "/"
                               + params["model_base"] + "-" + str(i_epoch) + "_" + str(i_batch),
                               )

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
