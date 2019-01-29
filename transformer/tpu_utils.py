# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions specific to running TensorFlow on TPUs."""

import tensorflow as tf

# "local" is a magic word in the TPU cluster resolver; it informs the resolver
# to use the local CPU as the compute device. This is useful for testing and
# debugging; the code flow is ostensibly identical, but without the need to
# actually have a TPU on the other end.
LOCAL = "local"


def embedding_matmul(embedding_table, values, mask, name="embedding_matmul"):
    """Performs embedding lookup via a matmul.

    The matrix to be multiplied by the embedding table Tensor is constructed
    via an implementation of scatter based on broadcasting embedding indices
    and performing an equality comparison against a broadcasted
    range(num_embedding_table_rows). All masked positions will produce an
    embedding vector of zeros.

    Args:
      embedding_table: Tensor of embedding table.
        Rank 2 (table_size x embedding dim)
      values: Tensor of embedding indices. Rank 2 (batch x n_indices)
      mask: Tensor of mask / weights. Rank 2 (batch x n_indices)
      name: Optional name scope for created ops

    Returns:
      Rank 3 tensor of embedding vectors.
    """

    with tf.name_scope(name):
        n_embeddings = embedding_table.get_shape().as_list()[0]
        batch_size, padded_size = values.shape.as_list()

        emb_idcs = tf.tile(
            tf.reshape(values, (batch_size, padded_size, 1)), (1, 1, n_embeddings))
        emb_weights = tf.tile(
            tf.reshape(mask, (batch_size, padded_size, 1)), (1, 1, n_embeddings))
        col_idcs = tf.tile(
            tf.reshape(tf.range(n_embeddings), (1, 1, n_embeddings)),
            (batch_size, padded_size, 1))
        one_hot = tf.where(
            tf.equal(emb_idcs, col_idcs), emb_weights,
            tf.zeros((batch_size, padded_size, n_embeddings)))

        return tf.tensordot(one_hot, embedding_table, 1)
