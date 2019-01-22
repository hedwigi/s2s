import jpype
from config import LIB_PARAMS


class PreprocessUtil:

    hanlp = None

    @staticmethod
    def init():
        # hanlp
        jarpath = LIB_PARAMS['jarpath']
        properties_path = LIB_PARAMS['properties_path']
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s:%s" % (jarpath, properties_path))
        PreprocessUtil.hanlp = jpype.JClass('com.hankcs.hanlp.HanLP')
        res = PreprocessUtil.hanlp.segment("")

    @staticmethod
    def words2idseq(raw_sentence, source_vocab2id):
        """

        :param raw_sentence:
        :param target_vocab2id:
        :return:
        """
        # tokens = [seg.word for seg in PreprocessUtil.hanlp.segment(raw_sentence)]
        tokens = raw_sentence.split()
        return [source_vocab2id[tok] if tok in source_vocab2id else "<UNK>" for tok in tokens]

    @staticmethod
    def idseq2words(idseq, id2target_vocab):
        """

        :param idseq:
        :param id2target_vocab:
        :return:
        """
        return " ".join([id2target_vocab[id] for id in idseq])

    @staticmethod
    def tokenize(line):
        """

        :param line:
        :return:
        """
        return [seg.word for seg in PreprocessUtil.hanlp.segment(line.strip())]

    @staticmethod
    def tokenize_with_pos(line):
        """

        :param line:
        :return:
        """
        token_pos = [(seg.word, seg.nature.toString()) for seg in PreprocessUtil.hanlp.segment(line.strip())]
        tokens = [t for t, p in token_pos]
        pos = [p for t, p in token_pos]
        return tokens, pos

    @staticmethod
    def preprocess(tokens, postags=None, max_len_emoji=5):

        if not postags:
            postags = ["" for t in tokens]

        emoji = []
        new_tokens = []
        emoji_pos = []
        new_postags = []
        for i in range(len(tokens)):
            if tokens[i] == "[" and len(emoji) == 0:
                emoji.append(tokens[i])
                emoji_pos.append(postags[i])
            elif tokens[i] == "]" and 0 < len(emoji):
                if len(emoji) <= max_len_emoji - 1:
                    emoji.append(tokens[i])
                    new_tokens.append("".join(emoji))
                    emoji = []

                    new_postags.append("xx")
                    emoji_pos = []

                # if len > max_len, not considered an emoji
                else:
                    new_tokens += emoji + [tokens[i]]
                    emoji = []

                    new_postags += emoji_pos + [postags[i]]
                    emoji_pos = []
            elif len(emoji) > 0:
                emoji.append(tokens[i])
                emoji_pos.append(postags[i])
            else:
                new_tokens.append(tokens[i])
                new_postags.append(postags[i])

        if len(emoji) > 0:
            new_tokens += emoji
            new_postags += emoji_pos

        assert len(new_tokens) == len(new_postags)
        return new_tokens, new_postags

    @staticmethod
    def ngrams(tokens, ngram_size):
        ngrams = []
        for win in range(1, ngram_size + 1):
            for i in range(len(tokens) - win + 1):
                ngrams.append("".join(tokens[i: i + win]))
        return ngrams

    @staticmethod
    def sortby_len_rewrite(pathx, pathy, pathout_x, pathout_y):
        with open(pathx, "r") as fin:
            all_x = fin.readlines()

        with open(pathy, "r") as fin:
            all_y = fin.readlines()

        # Sort data by source length
        sorted_pairs = sorted(list(zip(all_x, all_y)), key=lambda xy: len(xy[0].strip().split()))
        with open(pathout_x, "w") as fx:
            with open(pathout_y, "w") as fy:
                for x, y in sorted_pairs:
                    fx.write(x)
                    fy.write(y)


if __name__ == "__main__":
    PreprocessUtil.init()
    print(PreprocessUtil.preprocess(["a", "b", "[", "h","j","v","l", "]", "a", "b", "w", "]"], 5))
    print(PreprocessUtil.ngrams("你在敷衍我吗？", 3))
