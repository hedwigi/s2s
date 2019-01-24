import os


class VocabLoader(object):
    source_vocab2id = None
    target_vocab2id = None
    id2source_vocab = None
    id2target_vocab = None

    def __init__(self, path_x, path_y,
                 source_vocab_size, target_vocab_size,
                 default_vocab,
                 save_vocab,
                 source_vocab,
                 target_vocab):

        # ------ SOURCE VOCAB ------
        # sort vocab by frequency, keep top K in vocab
        source_vocab_count = {}
        self.load_vocab_from_text(path_x, source_vocab_count)
        source_vocab_count = sorted(source_vocab_count.items(), key=lambda v_c: v_c[1], reverse=True)[
                             :source_vocab_size - len(default_vocab)]
        # map vocab to id and vice-versa, keep special ids for default vocab
        self.source_vocab2id = {v_c[0]: i + len(default_vocab) for i, v_c in enumerate(source_vocab_count)}
        self.source_vocab2id.update(default_vocab)
        self.id2source_vocab = {i: v for v, i in self.source_vocab2id.items()}

        # ------ TARGET VOCAB ------
        target_vocab_count = {}
        self.load_vocab_from_text(path_y, target_vocab_count)
        target_vocab_count = sorted(target_vocab_count.items(), key=lambda v_c: v_c[1], reverse=True)[
                             :target_vocab_size - len(default_vocab)]
        self.target_vocab2id = {v_c[0]: i + len(default_vocab) for i, v_c in enumerate(target_vocab_count)}
        self.target_vocab2id.update(default_vocab)
        self.id2target_vocab = {i: v for v, i in self.target_vocab2id.items()}

        # write vocab to file
        source_vocab_count = dict(source_vocab_count)
        target_vocab_count = dict(target_vocab_count)
        if save_vocab:
            with open(source_vocab, "w") as fs:
                for v, i in sorted(self.source_vocab2id.items(), key=lambda v_i: v_i[1]):
                    count = source_vocab_count[v] if v in source_vocab_count else None
                    fs.write(v + " " + str(count) + "\n")

            with open(target_vocab, "w") as ft:
                for v, i in sorted(self.target_vocab2id.items(), key=lambda v_i: v_i[1]):
                    count = target_vocab_count[v] if v in target_vocab_count else None
                    ft.write(v + " " + str(count) + "\n")

    def get_vocab2id(self):
        return self.source_vocab2id, self.target_vocab2id

    def get_id2vocab(self):
        return self.id2source_vocab, self.id2target_vocab

    def load_vocab_from_text(self, pathfile, vocab_count):
        with open(pathfile, "r") as fx:
            line = fx.readline()
            while line:
                tokens = line.strip().split()
                for tok in tokens:
                    if tok not in vocab_count:
                        vocab_count[tok] = 0
                    vocab_count[tok] += 1
                line = fx.readline()

    @staticmethod
    def load_vocab2id(path_vocab):
        vocab2id = {}
        id2vocab = {}
        with open(path_vocab, "r") as fv:
            i = 0
            for l in fv:
                w, _ = l.strip().split(" ")
                vocab2id[w] = i
                id2vocab[i] = w
                i += 1
        return vocab2id, id2vocab
