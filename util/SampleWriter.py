import os


class SampleWriter:
    id2target_vocab = None
    id2source_vocab = None
    reverse_target = False
    end_id = None
    pad_id = None
    start_id = None
    dirdata = None

    def __init__(self, id2target_vocab, id2source_vocab, end_id, pad_id, start_id, reverse_target, dirdata):
        self.id2source_vocab = id2source_vocab
        self.id2target_vocab = id2target_vocab
        self.end_id = end_id
        self.pad_id = pad_id
        self.start_id = start_id
        self.reverse_target = reverse_target
        self.dirdata = dirdata

    def show_inference_samples(self, source_batch, target_batch, infer_batch, n_sample):
        for i in range(n_sample):
            source_words = self.__ids2words(source_batch[i], self.id2source_vocab)
            infer_words = self.__ids2words(infer_batch[i], self.id2target_vocab)
            target_words = self.__ids2words(target_batch[i], self.id2target_vocab)
            print("\tSource: %s,\tTarget: %s,\tInfer: %s" %
                  (" ".join(source_words), " ".join(target_words), " ".join(infer_words)))

    def write2file_inference_results(self, fout, valid_source_batch, infer_batch_logits):
        for i in range(len(valid_source_batch)):
            source_words = self.__ids2words(valid_source_batch[i], self.id2source_vocab)
            infer_words = self.__ids2words(infer_batch_logits[i], self.id2target_vocab)
            fout.write("Source: %s,\tInfer: %s\n" % (" ".join(source_words), " ".join(infer_words)))

    def valid_infer_filename(self, model_name, i_epoch):
        return os.path.join(self.dirdata, model_name + "_infer_results.ep" + str(i_epoch))

    def __ids2words(self, seq_ids, id2vocab):
        words = []
        for tok in seq_ids:
            words.append(id2vocab[tok])
            if not self.reverse_target and tok == self.end_id or tok == self.pad_id:
                break
            if self.reverse_target and tok == self.start_id:
                break
        return words
