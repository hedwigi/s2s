"""
Created on 2019/1/28

@author: wangyuqian
"""
from util.TransformerUtil import *
tf.set_random_seed(1)


class TransformerKyuGraph:
    def __init__(self, params, is_training=True):
        self.source_input = tf.placeholder(tf.int32, shape=(None, None), name="source_input")
        self.target = tf.placeholder(tf.int32, shape=(None, None), name="target")
        self.source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
        # if reverse_target, length doesn't include <S> at the end
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")

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
        self.train_acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.target)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.train_acc)

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