import re
from util.PreprocessUtil import PreprocessUtil


class PairDataWriter:

    @staticmethod
    def init():
        PreprocessUtil.init()

    @staticmethod
    def write_pairs(infile, outfile_source, outfile_target, params):
        """

        :param infile:
        :param outfile_source:
        :param outfile_target:
        :param params:
        :return:
        """
        iter = PairDataWriter.dialog_iterator(infile)

        with open(outfile_source, "w") as fs:
            with open(outfile_target, "w") as ft:
                dialog = iter.__next__()
                nd = 0
                finish = False
                while dialog:
                    for i in range(0, len(dialog), 2):
                        if i + 1 < len(dialog):
                            s = PreprocessUtil.preprocess(PreprocessUtil.tokenize(dialog[i].strip()), params["max_len_emoji"])
                            t = PreprocessUtil.preprocess(PreprocessUtil.tokenize(dialog[i + 1].strip()), params["max_len_emoji"])
                            if params["min_len_utterance"] <= len(s) <= params["max_len_utterance"]\
                                and params["min_len_utterance"] <= len(t) <= params["max_len_utterance"]:
                                nd += 1
                                if nd <= params["max_num_data"]:
                                    fs.write(" ".join(s) + "\n")
                                    ft.write(" ".join(t) + "\n")
                                else:
                                    finish = True
                    if finish:
                        break
                    dialog = iter.__next__()

    @staticmethod
    def dialog_iterator(file):
        with open(file, "r") as fin:
            dialog = []
            line = fin.readline()
            while line:
                if not re.match("\s*$", line):
                    dialog.append(line)
                else:
                    yield dialog
                    dialog = []
                line = fin.readline()
            yield dialog
        yield None


if __name__ == "__main__":

    infile = "/Users/wangyuqian/work/data/multi_1_4.4_100w.data"
    # infile = "ttt"
    outfile_source = "train_x"
    outfile_target = "train_y"

    params = {
        "max_num_data": 1000000,
        "min_len_utterance": 2,
        "max_len_utterance": 15,
        "max_len_emoji": 5,
    }

    PairDataWriter.init()
    PairDataWriter.write_pairs(infile, outfile_source, outfile_target, params)
