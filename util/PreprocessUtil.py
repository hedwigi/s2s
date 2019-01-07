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
        tokens = [seg.word for seg in PreprocessUtil.hanlp.segment(raw_sentence)]
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
    def preprocess(tokens, max_len_emoji):
        """

        :param tokens: list of str
        :return:
        """
        emoji = []
        new_tokens = []
        for i in range(len(tokens)):
            if tokens[i] == "[" and len(emoji) == 0:
                emoji.append("[")
            elif tokens[i] == "]" and 0 < len(emoji):
                if len(emoji) <= max_len_emoji - 1:
                    emoji.append("]")
                    new_tokens.append("".join(emoji))
                    emoji = []
                # if len > max_len, not considered an emoji
                else:
                    new_tokens += emoji + ["]"]
                    emoji = []
            elif len(emoji) > 0:
                emoji.append(tokens[i])
            else:
                new_tokens.append(tokens[i])

        if len(emoji) > 0:
            new_tokens += emoji

        return new_tokens


if __name__ == "__main__":
    print(PreprocessUtil.preprocess(["a", "b", "[", "h","j","v","l", "]", "a", "b", "w", "]"], 5))
