import numpy as np


class BatchIterator(object):
    source_path = None
    target_path = None
    source_iter = None
    target_iter = None

    batch_source = None
    batch_target = None
    batch_src_length = None
    batch_tgt_length = None

    reverse_target = False

    def __init__(self, source_path, target_path,
                 source_vocab2id, target_vocab2id,
                 start_id, end_id, unk_id, pad_id,
                 reverse_target=False):

        self.source_path = source_path
        self.target_path = target_path
        self.start_id, self.end_id, self.unk_id, self.pad_id = start_id, end_id, unk_id, pad_id
        self.source_vocab2id, self.target_vocab2id = source_vocab2id, target_vocab2id
        self.reverse_target = reverse_target

        self.source_iter = open(source_path, "r")
        self.target_iter = open(target_path, "r")

    def has_next(self, batch_size):
        source = self.source_iter.readline()
        target = self.target_iter.readline()

        self.batch_source = []
        self.batch_target = []
        self.batch_src_length = []
        self.batch_tgt_length = []
        i = 0
        while source and target:
            i += 1
            if i > batch_size:
                break

            # target_ids:<S>..<UNK>..<EOS>
            source = source.strip().split(" ")
            target = target.strip().split(" ")
            source = [self.source_vocab2id[w] if w in self.source_vocab2id else self.unk_id for w in source]
            target = [self.target_vocab2id[w] if w in self.target_vocab2id else self.unk_id for w in target] + [self.end_id]
            if self.reverse_target:
                target.reverse()

            self.batch_source.append(source)
            self.batch_target.append(target)
            self.batch_src_length.append(len(source))
            self.batch_tgt_length.append(len(target))

            source = self.source_iter.readline()
            target = self.target_iter.readline()

        if len(self.batch_source) == batch_size:
            add_start = True if self.reverse_target else False
            self.batch_source = self.padding(self.batch_source)
            self.batch_target = self.padding(self.batch_target, add_start)
            return True
        return False

    def next_batch(self, batch_size):
        return self.batch_source, self.batch_target, self.batch_src_length, self.batch_tgt_length

    def reset(self):
        self.source_iter.close()
        self.target_iter.close()
        self.source_iter = open(self.source_path, "r")
        self.target_iter = open(self.target_path, "r")

    def padding(self, l, add_start=False):
        max_len = max([len(sent) for sent in l])

        if add_start:
            return np.array([sent + [self.start_id] + [self.pad_id] * (max_len - len(sent) - 1)
                             if max_len > len(sent) else sent
                             for sent in l
                             ])
        return np.array([sent + [self.pad_id] * (max_len - len(sent)) for sent in l])