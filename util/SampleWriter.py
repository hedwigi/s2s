class SampleWriter:
    id2target_vocab = None
    id2source_vocab = None

    def __init__(self, id2target_vocab, id2source_vocab, end_id, pad_id):
        self.id2source_vocab = id2source_vocab
        self.id2target_vocab = id2target_vocab
        self.end_id = end_id
        self.pad_id = pad_id

    def write_inference_samples(self, source_batch, infer_batch, n_sample):
        """

        :param source_batch:
        :param infer_batch:
        :return:
        """
        for i in range(n_sample):
            source_words = self.__ids2words(source_batch[i], self.id2source_vocab)
            infer_words = self.__ids2words(infer_batch[i], self.id2target_vocab)
            print("\tSource: %s,\tInfer: %s" % (" ".join(source_words), " ".join(infer_words)))

    def __ids2words(self, seq_ids, id2vocab):
        words = []
        for tok in seq_ids:
            words.append(id2vocab[tok])
            if tok == self.end_id or tok == self.pad_id:
                break
        return words
