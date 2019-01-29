"""
Created on 2019/1/28

@author: wangyuqian
"""
import tensorflow as tf
from util.Monitor import Monitor


class Seq2SeqGraph:

    def __init__(self, params_common, params_model, is_training):
        self.params_common = params_common
        self.params_model = params_model

        # *************** PLACEHOLDER & INPUT ***************
        # [batch_size, sequence_len]
        self.source_input = tf.placeholder(tf.int32, [None, None], name="source_input")
        self.target = tf.placeholder(tf.int32, [None, None], name="target")

        self.source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
        # if reverse_target, length doesn't include <S> at the end
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")

        # 获取max target len
        max_target_len = tf.reduce_max(self.target_sequence_length)

        # 获取可变的batch_size
        batch_size = tf.shape(self.source_input)[0]

        if params_common["reverse_target"]:
            # target input: <EOS> 4 3 2 <S> <PAD>, <EOS> 6 5 4 3 2,
            target_input = self.target
            # target output: 3 2 1 <S> <PAD> <S>, 6 5 4 3 2 <S>
            # target seq len: 4,             6
            first_slices = tf.strided_slice(target_input, [0,1], [batch_size, max_target_len], [1, 1])
            self.target_output = tf.concat([first_slices, tf.fill([batch_size, 1], params_common["start_id"])], 1)
        else:
            # target output: 1 2 3 <EOS>
            self.target_output = self.target
            # target input: <S> 1 2 3
            after_slice = tf.strided_slice(self.target_output, [0, 0], [batch_size, -1], [1, 1])
            target_input = tf.concat([tf.fill([batch_size, 1], params_common["start_id"]), after_slice], 1)

        # *************** GRAPH ****************
        if not is_training:
            params_model["keep_prob"] = 1.0

        # ------ RNN Encoder ------
        # TODO change to independent embedding
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            enc_embeddings = tf.get_variable("input_embedding",
                                         initializer=tf.random_uniform([params_common["source_vocab_size"],
                                                                        params_model["encoding_embedding_size"]]))
            # (uni-rnn)
            # list of separated rnn cells
            # rnn_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params["rnn_size"]), params["keep_prob"])

            # stack n layers together
            # stacked_cells = tf.contrib.rnn.MultiRNNCell(l_dropped_out_rnn_cell)
            # unroll rnn_cell instance, output是通过主动提供输入tok得到的
            # _, encoder_state = tf.nn.dynamic_rnn(rnn_cell, embed, dtype=tf.float32)

            # (bi-rnn)
            input = tf.nn.embedding_lookup(enc_embeddings, self.source_input)
            encoder_states = []
            for _ in range(params_model["num_layers"]):
                with tf.variable_scope(None, default_name="stacked_bilstm"):
                    fw_rnn_cell = tf.contrib.rnn.LSTMCell(params_model["rnn_size"] / 2)
                    fw_dropped_out_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, params_model["keep_prob"])

                    bw_rnn_cell = tf.contrib.rnn.LSTMCell(params_model["rnn_size"] / 2)
                    bw_dropped_out_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, params_model["keep_prob"])

                    # outputs = tuple(fw_out, bw_out)
                    # state = (fw_state, bw_state), fw_state = (c, h)
                    outputs, state = tf.nn.bidirectional_dynamic_rnn(
                        fw_dropped_out_rnn_cell, bw_dropped_out_rnn_cell, input, self.source_sequence_length, dtype=tf.float32
                    )

                    # update
                    input = tf.concat(outputs, 2)
                    states = tf.concat([state[0], state[1]], axis=2)
                    encoder_state = tf.nn.rnn_cell.LSTMStateTuple(states[0], states[1])
                    encoder_states.append(encoder_state)
            # encoder_states = tuple(encoder_states)  # if no attention
            # [batch_size, max_time, num_units]
            encoder_outputs = input

        # ------ RNN Decoder -------
        # reuse: shared rnn cells
        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):

            # Create an attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                params_model["rnn_size"], encoder_outputs,
                memory_sequence_length=self.source_sequence_length)

            dec_embeddings = tf.get_variable("output_embedding",
                                             initializer=tf.random_uniform([params_common["target_vocab_size"],
                                                                            params_model["decoding_embedding_size"]]))
            # dec_embeddings = tf.Variable(tf.random_uniform([params_common["target_vocab_size"],
            #                                                     params_model["decoding_embedding_size"]]),
            #                              name="output_embedding")
            dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, target_input)

            l_dec_rnn_cell = [tf.contrib.rnn.LSTMCell(params_model["rnn_size"]) for i in
                              range(params_model["num_layers"])]
            dec_stacked_cells = tf.contrib.rnn.MultiRNNCell(l_dec_rnn_cell)
            dec_stacked_att_cells = tf.contrib.seq2seq.AttentionWrapper(
                dec_stacked_cells, attention_mechanism,
                attention_layer_size=params_model["rnn_size"])

            # --- Train phase ---
            dec_train_cells = tf.contrib.rnn.DropoutWrapper(dec_stacked_att_cells,
                                                            output_keep_prob=params_model["keep_prob"])
            output_layer = tf.layers.Dense(params_common["target_vocab_size"])

            # dynamic_rnn只能使用提供的input得到output，（helper + decoder + dynamic_decode）可以自定义得到output的方式
            # 由helper决定decoder的input。此处dec_embed是true label的输入tok
            helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, self.target_sequence_length)

            # 核心decoder，使用helper的input和rnn_cell，以及输出层，返回单次的RNN output
            decoder_train = tf.contrib.seq2seq.BasicDecoder(dec_train_cells, helper,
                                                            dec_train_cells.zero_state(dtype=tf.float32,
                                                                                       batch_size=batch_size),
                                                            output_layer)

            # 使用核心decoder，提供用来unroll的大循环
            self.decoder_train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=max_target_len)

            # --- Infer phase ---
            infer_start = params_common["end_id"] if params_common["reverse_target"] else params_common["start_id"]
            infer_end = params_common["start_id"] if params_common["reverse_target"] else params_common["end_id"]
            gd_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                 tf.fill([batch_size], infer_start),
                                                                 infer_end)
            decoder_infer = tf.contrib.seq2seq.BasicDecoder(dec_stacked_att_cells, gd_helper,
                                                            dec_train_cells.zero_state(dtype=tf.float32,
                                                                                       batch_size=batch_size),
                                                            output_layer)
            self.decoder_infer_outputs, _, self.infer_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder_infer,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=max_target_len)

            # ------ FORWARD -------
            # TODO: with same value, why need identity?
            self.training_logits = tf.identity(self.decoder_train_outputs.rnn_output, name="logits")
            self.inference_sample_id = tf.identity(self.decoder_infer_outputs.sample_id, name="predictions")

        vars = tf.trainable_variables()
        Monitor.print_params(vars)
