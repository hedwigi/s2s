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
