import os
import time
import tensorflow as tf
from config import params_common, params_models
from Seq2Seq import Seq2Seq
from entity.BatchIterator import BatchIterator
from util.SampleWriter import SampleWriter
from util.VocabLoader import VocabLoader
from util.PreprocessUtil import PreprocessUtil
from Transformer import Transformer


# **** PARAM ****
dirdata = os.path.join(os.path.dirname(__file__), params_common["datadir"])
path_train_x_orig = os.path.join(dirdata, "train_x_" + params_common["data_suffix"])
path_train_y_orig = os.path.join(dirdata, "train_y_" + params_common["data_suffix"])
path_valid_x_orig = os.path.join(dirdata, "valid_x_" + params_common["data_suffix"])
path_valid_y_orig = os.path.join(dirdata, "valid_y_" + params_common["data_suffix"])

# **** SORT BY LENGTH ****
path_train_x = path_train_x_orig + ".sorted"
path_train_y = path_train_y_orig + ".sorted"
path_valid_x = path_valid_x_orig + ".sorted"
path_valid_y = path_valid_y_orig + ".sorted"

# **** DATA PROCESS ****
default_vocab = {"<PAD>": params_common["pad_id"],
                 "<S>": params_common["start_id"],
                 "<EOS>": params_common["end_id"],
                 "<UNK>": params_common["unk_id"]}

# **** CONSTRUCT/LOAD VOCAB ****
if params_common["mode"] == "train":
    PreprocessUtil.sortby_len_rewrite(path_train_x_orig, path_train_y_orig,
                                      path_train_x, path_train_y)
    PreprocessUtil.sortby_len_rewrite(path_valid_x_orig, path_valid_y_orig,
                                      path_valid_x, path_valid_y)

    vocab_loader = VocabLoader(path_train_x, path_train_y,
                               params_common["source_vocab_size"], params_common["target_vocab_size"],
                               default_vocab,
                               True,
                               params_common["source_vocab"],
                               params_common["target_vocab"])

    source_vocab2id, target_vocab2id = vocab_loader.get_vocab2id()
    id2source_vocab, id2target_vocab = vocab_loader.get_id2vocab()

else:
    # inference
    source_vocab2id, target_vocab2id = VocabLoader.load_vocab2id(params_common["source_vocab"])
    id2source_vocab, id2target_vocab = VocabLoader.load_vocab2id(params_common["target_vocab"])

# update params
params_common["source_vocab_size"] = min(len(source_vocab2id), params_common["source_vocab_size"])
params_common["target_vocab_size"] = min(len(id2target_vocab), params_common["target_vocab_size"])


if __name__ == "__main__":

    sess = tf.Session()
    options = None
    run_metadata = None
    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    train_timeline_fname = 'timeline_01.json'
    valid_timeline_fname = "timeline_infer_1s"

    if params_common["model_name"][0] == "Transformer":
        model_params = params_models[params_common["model_name"][0]][params_common["model_name"][1]]
        model_params.update(vocab_size=max(params_common["source_vocab_size"], params_common["target_vocab_size"]))
        model = Transformer(params_common, model_params)
    else:
        model_params = params_models[params_common["model_name"][0]]
        model = Seq2Seq(params_common, model_params)
    print(model_params)

    if params_common["mode"] == "train":

        train_iter = BatchIterator(path_train_x, path_train_y, source_vocab2id, target_vocab2id,
                                   params_common["start_id"], params_common["end_id"], params_common["unk_id"], params_common["pad_id"],
                                   params_common["reverse_target"])

        valid_iter = BatchIterator(path_valid_x, path_valid_y, source_vocab2id, target_vocab2id,
                                   params_common["start_id"], params_common["end_id"], params_common["unk_id"], params_common["pad_id"],
                                   params_common["reverse_target"])

        sample_writer = SampleWriter(id2target_vocab, id2source_vocab,
                                     params_common["end_id"], params_common["pad_id"], params_common["start_id"],
                                     params_common["reverse_target"], params_common["results_dir"])

        model.train(sess, train_iter, valid_iter, sample_writer,
                    options, run_metadata, train_timeline_fname)

    elif params_common["mode"] == "single":
        raw_question = "明天 开始 军训 了"
        question_in_id = PreprocessUtil.words2idseq(raw_question, source_vocab2id)
        response_in_id = model.infer(sess, question_in_id,
                                     options, run_metadata, valid_timeline_fname)
        response = PreprocessUtil.idseq2words(response_in_id, id2target_vocab)
        print("Q: %s\n" % raw_question)
        print("R: %s\n" % response)

    elif params_common["mode"] == "valid_batch":
        file = "data/valid_batch_x"
        start = time.clock()
        with open(file, "r") as fin:
            count = 0
            for l in fin:
                count += 1
                raw_question = l.strip()
                question_in_id = PreprocessUtil.words2idseq(raw_question, source_vocab2id)
                response_in_id = model.infer(sess, question_in_id,
                                             options, run_metadata, valid_timeline_fname)
                response = PreprocessUtil.idseq2words(response_in_id, id2target_vocab)
                print("Q: %s\n" % raw_question)
                print("R: %s\n" % response)
        print("Avg inference time %f s over %d samples" % ((time.clock() - start)/count, count))
    else:
        pass
