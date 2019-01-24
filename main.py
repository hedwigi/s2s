import os
import time
import tensorflow as tf
from config import params
from Seq2Seq import Seq2Seq
from entity.BatchIterator import BatchIterator
from util.SampleWriter import SampleWriter
from util.VocabLoader import VocabLoader
from util.PreprocessUtil import PreprocessUtil

mode = "train"

# **** PARAM ****
dirdata = os.path.join(os.path.dirname(__file__), "data")
path_train_x_orig = os.path.join(dirdata, "train_x_small")
path_train_y_orig = os.path.join(dirdata, "train_y_small")
path_valid_x_orig = os.path.join(dirdata, "valid_x_small")
path_valid_y_orig = os.path.join(dirdata, "valid_y_small")

# **** SORT BY LENGTH ****
path_train_x = path_train_x_orig + ".sorted"
path_train_y = path_train_y_orig + ".sorted"
path_valid_x = path_valid_x_orig + ".sorted"
path_valid_y = path_valid_y_orig + ".sorted"

# **** DATA PROCESS ****
default_vocab = {"<PAD>": params["pad_id"],
                 "<S>": params["start_id"],
                 "<EOS>": params["end_id"],
                 "<UNK>": params["unk_id"]}

# **** CONSTRUCT/LOAD VOCAB ****
if mode == "train":
    PreprocessUtil.sortby_len_rewrite(path_train_x_orig, path_train_y_orig,
                                      path_train_x, path_train_y)
    PreprocessUtil.sortby_len_rewrite(path_valid_x_orig, path_valid_y_orig,
                                      path_valid_x, path_valid_y)

    vocab_loader = VocabLoader(path_train_x, path_train_y,
                               params["source_vocab_size"], params["target_vocab_size"],
                               default_vocab,
                               True,
                               params["source_vocab"],
                               params["target_vocab"])

    source_vocab2id, target_vocab2id = vocab_loader.get_vocab2id()
    id2source_vocab, id2target_vocab = vocab_loader.get_id2vocab()

else:
    # inference
    source_vocab2id, target_vocab2id = VocabLoader.load_vocab2id(params["source_vocab"])
    id2source_vocab, id2target_vocab = VocabLoader.load_vocab2id(params["target_vocab"])
    params["keep_prob"] = 1.0

# update params
params["source_vocab_size"] = min(len(source_vocab2id), params["source_vocab_size"])
params["target_vocab_size"] = min(len(id2target_vocab), params["target_vocab_size"])


if __name__ == "__main__":

    sess = tf.Session()
    options = None
    run_metadata = None
    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    train_timeline_fname = 'timeline_01.json'
    valid_timeline_fname = "timeline_infer_1s"
    model = Seq2Seq(params)

    # print("PARAMS:\n%s" % params)
    # bilstm_cell_params_1 = 4 * (params["rnn_size"] / 2) \
    #                        * (params["rnn_size"] / 2 + params["encoding_embedding_size"] + 1)
    # bilstm_cell_params_o = 4 * (params["rnn_size"] / 2) \
    #                        * (params["rnn_size"] / 2 + params["rnn_size"] + 1)
    # rnn_cell_params_1 = 4 * params["rnn_size"] * (params["rnn_size"] + params["decoding_embedding_size"] + 1)
    # rnn_cell_params_o = 4 * params["rnn_size"] * (params["rnn_size"] + params["rnn_size"] + 1)
    #
    # total_bilstm_size = bilstm_cell_params_1 * 2 + (params["num_layers"] - 1) * bilstm_cell_params_o * 2
    # total_rnn_size = rnn_cell_params_1 + (params["num_layers"] - 1) * rnn_cell_params_o
    # print("BiLSTM params size: %d" % total_bilstm_size)
    # print("RNN params size without Attention: %d" % total_rnn_size)
    # print("Total params size: %d" % (total_bilstm_size + total_rnn_size))

    if mode == "train":

        train_iter = BatchIterator(path_train_x, path_train_y, source_vocab2id, target_vocab2id,
                                   params["start_id"], params["end_id"], params["unk_id"], params["pad_id"],
                                   params["reverse_target"])

        valid_iter = BatchIterator(path_valid_x, path_valid_y, source_vocab2id, target_vocab2id,
                                   params["start_id"], params["end_id"], params["unk_id"], params["pad_id"],
                                   params["reverse_target"])

        sample_writer = SampleWriter(id2target_vocab, id2source_vocab,
                                     params["end_id"], params["pad_id"], params["start_id"],
                                     params["reverse_target"], dirdata)

        model.train(sess, train_iter, valid_iter, params, sample_writer,
                    options, run_metadata, train_timeline_fname)

    elif mode == "single":
        raw_question = "明天 开始 军训 了"
        question_in_id = PreprocessUtil.words2idseq(raw_question, source_vocab2id)
        response_in_id = model.infer(sess, question_in_id, params,
                                     options, run_metadata, valid_timeline_fname)
        response = PreprocessUtil.idseq2words(response_in_id, id2target_vocab)
        print("Q: %s\n" % raw_question)
        print("R: %s\n" % response)

    elif mode == "valid_batch":
        file = "data/valid_batch_x"
        start = time.clock()
        with open(file, "r") as fin:
            count = 0
            for l in fin:
                count += 1
                raw_question = l.strip()
                question_in_id = PreprocessUtil.words2idseq(raw_question, source_vocab2id)
                response_in_id = model.infer(sess, question_in_id, params,
                                             options, run_metadata, valid_timeline_fname)
                response = PreprocessUtil.idseq2words(response_in_id, id2target_vocab)
                print("Q: %s\n" % raw_question)
                print("R: %s\n" % response)
        print("Avg inference time %f s over %d samples" % ((time.clock() - start)/count, count))
    else:
        pass
