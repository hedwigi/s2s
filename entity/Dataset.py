import numpy as np
from config import params


class Dataset(object):

    source_ids = None
    target_ids = None
    start = None

    def __init__(self, source, target,
                 source_vocab2id, target_vocab2id):
        """

        :param source: [[tok, tok], ...]
        :param target:
        :param params:
        :param source_vocab2id:
        :param target_vocab2id:
        """
        self.source_ids = []
        self.target_ids = []
        self.start = 0

        # target_ids:<S>..<UNK>..<EOS>

        # sort data by source_ids length
        data = sorted(list(zip(source, target)), key=lambda st: len(st[0]))
        for s, t in data:
            self.source_ids.append([source_vocab2id[w] if w in source_vocab2id else params["unk_id"] for w in s])
            self.target_ids.append([params["start_id"]] \
                                   + [target_vocab2id[w] if w in target_vocab2id else params["unk_id"] for w in t] \
                                   + [params["end_id"]])

    def has_next(self, batch_size):
        return self.start + batch_size <= len(self.source_ids)

    def next_batch(self, batch_size):
        # pad source_ids and target_ids batch
        end = self.start + batch_size
        if end <= len(self.source_ids):

            source_batch = self.source_ids[self.start: end]
            target_batch = self.target_ids[self.start: end]
            source_lengths = [len(sent) for sent in source_batch]
            target_lengths = [len(sent) for sent in target_batch]

            self.start += batch_size
            return self.padding(source_batch), self.padding(target_batch), \
                   source_lengths, target_lengths
        else:
            return None, None, None, None

    def reset(self):
        self.start = 0

    def padding(self, l):
        """

        :param l:
        :return:
        """
        max_len = max([len(sent) for sent in l])
        return np.array([sent + [params["pad_id"]] * (max_len - len(sent)) for sent in l])
