import os
import tensorflow as tf
from config import params
from Seq2Seq import Seq2Seq
from entity.Dataset import Dataset
from util.SampleWriter import SampleWriter
from util.DataLoader import DataLoader
from util.PreprocessUtil import PreprocessUtil


# **** PARAM ****
dirdata = os.path.join(os.path.dirname(__file__), "data")
path_train_x = os.path.join(dirdata, "train_x_small")
path_train_y = os.path.join(dirdata, "train_y_small")
path_valid_x = os.path.join(dirdata, "valid_x_small")
path_valid_y = os.path.join(dirdata, "valid_y_small")

default_vocab = {"<PAD>": params["pad_id"],
                  "<S>": params["start_id"],
                  "<EOS>": params["end_id"],
                  "<UNK>": params["unk_id"]}

train_loader = DataLoader(path_train_x, path_train_y,
                          params["source_vocab_size"], params["target_vocab_size"],
                          default_vocab,
                          True,
                          "train")

train_x, train_y = train_loader.get_x_y()
valid_x = DataLoader.load_without_vocab(path_valid_x)
valid_y = DataLoader.load_without_vocab(path_valid_y)
# train_x, train_y, valid_x, valid_y = train_loader.split_train_valid(params["valid_size"])

source_vocab2id, target_vocab2id = train_loader.get_vocab2id()

# update params
params["source_vocab_size"] = min(len(source_vocab2id), params["source_vocab_size"])
params["target_vocab_size"] = min(len(target_vocab2id), params["target_vocab_size"])
id2source_vocab, id2target_vocab = train_loader.get_id2vocab()

trainset = Dataset(train_x, train_y, source_vocab2id, target_vocab2id,
                   params["start_id"], params["end_id"], params["unk_id"], params["pad_id"],
                   params["reverse_target"])

validset = Dataset(valid_x, valid_y, source_vocab2id, target_vocab2id,
                   params["start_id"], params["end_id"], params["unk_id"], params["pad_id"],
                   params["reverse_target"])

sample_writer = SampleWriter(id2target_vocab, id2source_vocab,
                             params["end_id"], params["pad_id"], params["start_id"],
                             params["reverse_target"])


if __name__ == "__main__":

    mode = "train"

    sess = tf.Session()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    model = Seq2Seq(params)
    if mode == "train":
        print("PARAMS:\n%s" % params)
        bilstm_cell_params_1 = 4 * (params["rnn_size"] / 2) \
                               * (params["rnn_size"] / 2 + params["encoding_embedding_size"] + 1)
        bilstm_cell_params_o = 4 * (params["rnn_size"]/2) \
                               * (params["rnn_size"] / 2 + params["rnn_size"] + 1)
        rnn_cell_params_1 = 4 * params["rnn_size"] * (params["rnn_size"] + params["decoding_embedding_size"] + 1)
        rnn_cell_params_o = 4 * params["rnn_size"] * (params["rnn_size"] + params["rnn_size"] + 1)

        total_bilstm_size = bilstm_cell_params_1 * 2 + (params["num_layers"] - 1) * bilstm_cell_params_o * 2
        total_rnn_size = rnn_cell_params_1 + (params["num_layers"] - 1) * rnn_cell_params_o
        print("BiLSTM params size: %d" % total_bilstm_size)
        print("RNN params size: %d" % total_rnn_size)
        print("Total params size: %d" % (total_bilstm_size + total_rnn_size))
        model.train(sess, trainset, validset, params, sample_writer, options, run_metadata)

    elif mode == "single":
        raw_question = "they like pears , apples , and mangoes ."
        question_in_id = PreprocessUtil.words2idseq(raw_question, source_vocab2id)
        timeline_fname = "timeline_infer_1s"
        response_in_id = model.infer(sess, question_in_id, params, options, run_metadata, timeline_fname)
        response = PreprocessUtil.idseq2words(response_in_id, id2target_vocab)
        print("Q: %s\n" % raw_question)
        print("R: %s\n" % response)
