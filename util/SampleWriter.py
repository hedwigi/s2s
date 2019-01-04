class SampleWriter:
    id2target_vocab = None

    def __init__(self, id2target_vocab, end_id, pad_id):
        self.id2target_vocab = id2target_vocab
        self.end_id = end_id
        self.pad_id = pad_id

    def write_inference_samples(self, source_batch, infer_batch):
        """

        :param source_batch:
        :param infer_batch:
        :return:
        """
        for i in range(source_batch.shape[0]):
            source_words = self.__ids2words(source_batch[i])
            infer_words = self.__ids2words(infer_batch[i])
            print("\tSource: %s,\tInfer: %s" % (" ".join(source_words), " ".join(infer_words)))

    def __ids2words(self, seq_ids):
        words = []
        for tok in seq_ids:
            words.append(self.id2target_vocab[tok])
            if tok == self.end_id or tok == self.pad_id:
                break
        return words


