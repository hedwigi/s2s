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
path_train_x = os.path.join(dirdata, "train_x")
path_train_y = os.path.join(dirdata, "train_y")
path_test_x = os.path.join(dirdata, "test_x")
path_test_y = os.path.join(dirdata, "test_y")

default_vocab = {"<PAD>": params["pad_id"],
                  "<S>": params["start_id"],
                  "<EOS>": params["end_id"],
                  "<UNK>": params["unk_id"]}

train_loader = DataLoader(path_train_x, path_train_y,
                          params["source_vocab_size"], params["target_vocab_size"],
                          default_vocab,
                          True,
                          "train")

train_x, train_y, valid_x, valid_y = train_loader.split_train_valid(params["valid_size"])
source_vocab2id, target_vocab2id = train_loader.get_vocabs2id()
# update params
params["source_vocab_size"] = min(len(source_vocab2id), params["source_vocab_size"])
params["target_vocab_size"] = min(len(target_vocab2id), params["target_vocab_size"])
id2source_vocab, id2target_vocab = train_loader.get_id2vocab()

trainset = Dataset(train_x, train_y, source_vocab2id, target_vocab2id)
validset = Dataset(valid_x, valid_y, source_vocab2id, target_vocab2id)
sample_writer = SampleWriter(id2target_vocab, id2source_vocab, params["end_id"], params["pad_id"])

# test
# test_loader = DataLoader(path_test_x, path_test_y,
#                           params["source_vocab_size"], params["target_vocab_size"],
#                           default_vocab,
#                           True,
#                          "test")
# test_x, test_y, _, _ = test_loader.split_train_valid(0)


if __name__ == "__main__":

    mode = "train"

    sess = tf.Session()
    model = Seq2Seq(params)
    if mode == "train":
        model.train(sess, trainset, validset, params, sample_writer)

    elif mode == "single":
        raw_question = "你好吗"
        question_in_id = PreprocessUtil.words2idseq(raw_question, source_vocab2id)
        response_in_id = model.infer(sess, question_in_id, params)
        response = PreprocessUtil.idseq2words(response_in_id, id2target_vocab)
        print("Q: %s\n" % raw_question)
        print("R: %s\n" % response)
