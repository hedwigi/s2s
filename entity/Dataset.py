import numpy as np


class Dataset(object):

    source_ids = None
    target_ids = None
    start = None
    reverse_target = False
    start_id, end_id, unk_id, pad_id = None, None, None, None

    def __init__(self, source, target,
                 source_vocab2id, target_vocab2id,
                 start_id, end_id, unk_id, pad_id,
                 reverse_target=False):
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
        self.start_id, self.end_id, self.unk_id, self.pad_id = start_id, end_id, unk_id, pad_id
        self.reverse_target = reverse_target

        # target_ids:<S>..<UNK>..<EOS>

        # sort data by source_ids length
        data = sorted(list(zip(source, target)), key=lambda st: len(st[0]))
        for s, t in data:
            self.source_ids.append([source_vocab2id[w] if w in source_vocab2id else self.unk_id for w in s])
            target_id = [target_vocab2id[w] if w in target_vocab2id else self.unk_id for w in t] + [self.end_id]
            if self.reverse_target:
                target_id.reverse()
            self.target_ids.append(target_id)

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
            add_start = True if self.reverse_target else False
            return self.padding(source_batch), self.padding(target_batch, add_start), \
                   source_lengths, target_lengths
        else:
            return None, None, None, None

    def reset(self):
        self.start = 0

    def padding(self, l, add_start=False):
        max_len = max([len(sent) for sent in l])

        if add_start:
            return np.array([sent + [self.start_id] + [self.pad_id] * (max_len - len(sent) - 1)
                             if max_len > len(sent) else sent
                             for sent in l
                             ])
        return np.array([sent + [self.pad_id] * (max_len - len(sent)) for sent in l])
