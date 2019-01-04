import os


class DataLoader(object):
    all_x = None
    all_y = None
    source_vocab2id = None
    target_vocab2id = None
    id2target_vocab = None

    def __init__(self, path_x, path_y,
                 source_vocab_size, target_vocab_size,
                 default_vocab,
                 save_vocab,
                 vocab_name):

        self.all_x = []
        source_vocab_count = {}
        self.load_data_n_vocab(path_x, self.all_x, source_vocab_count)
        source_vocab_count = sorted(source_vocab_count.items(), key=lambda v_c: v_c[1], reverse=True)[:source_vocab_size]
        self.source_vocab2id = {v_c[0]: i + len(default_vocab) for i, v_c in enumerate(source_vocab_count)}
        self.source_vocab2id.update(default_vocab)

        self.all_y = []
        target_vocab_count = {}
        self.load_data_n_vocab(path_y, self.all_y, target_vocab_count)
        target_vocab_count = sorted(target_vocab_count.items(), key=lambda v_c: v_c[1], reverse=True)[:target_vocab_size]
        self.target_vocab2id = {v_c[0]: i + len(default_vocab) for i, v_c in enumerate(target_vocab_count)}
        self.target_vocab2id.update(default_vocab)
        self.id2target_vocab = {i: v for v, i in self.target_vocab2id.items()}

        if save_vocab:
            dir = os.path.dirname(path_x)
            with open(os.path.join(dir, vocab_name + ".source_vocab"), "w") as fs:
                for v, i in sorted(self.source_vocab2id.items(), key=lambda v_i: v_i[1]):
                    fs.write(v + "\n")

            with open(os.path.join(dir, vocab_name + ".target_vocab"), "w") as ft:
                for v, i in sorted(self.target_vocab2id.items(), key=lambda v_i: v_i[1]):
                    ft.write(v + "\n")

    def split_train_valid(self, valid_size):
        num_valid = int(len(self.all_x) * valid_size)
        train_x, valid_x = self.all_x[num_valid:], self.all_x[:num_valid]
        train_y, valid_y = self.all_y[num_valid:], self.all_y[:num_valid]
        print("train size: %d, valid size: %d" % (len(train_x), len(valid_x)))
        return train_x, train_y, valid_x, valid_y

    def get_vocabs2id(self):
        return self.source_vocab2id, self.target_vocab2id

    def get_id2target_vocab(self):
        return self.id2target_vocab

    def load_data_n_vocab(self, pathfile, data_list, vocab_count):
        """

        :param pathfile:
        :param data_list: [] -> [[tok, tok, ...], ]
        :param vocab_count:
        :param default_vocab:
        :return:
        """
        with open(pathfile, "r") as fx:
            line = fx.readline()
            while line:
                tokens = line.strip().split()
                for tok in tokens:
                    if tok not in vocab_count:
                        vocab_count[tok] = 0
                    vocab_count[tok] += 1
                data_list.append(tokens)
                line = fx.readline()
