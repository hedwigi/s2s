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

                """
                Q: c5 -> i - 5
                A: c4
                Q: c3
                A: c2
                Q: c1
                A: source -> i
                Q: target -> i + 1
                """

                while dialog:
                    for i in range(0, len(dialog), 2):
                        if i + 1 < len(dialog):
                            # valid context or not
                            valid_context = True
                            context_toks = []
                            context = dialog[max(0, i - params["max_context_size"]): i]
                            for j, c in enumerate(context):
                                c_toks, _ = PreprocessUtil.preprocess(PreprocessUtil.tokenize(c.strip()), max_len_emoji=params["max_len_emoji"])
                                if not params["min_len_utterance"] <= len(c_toks) <= params["max_len_utterance"]:
                                    valid_context = False
                                    break
                                else:
                                    context_toks += c_toks + ["<SEP>"]

                            if valid_context:
                                s, _ = PreprocessUtil.preprocess(PreprocessUtil.tokenize(dialog[i].strip()), max_len_emoji=params["max_len_emoji"])
                                t, _ = PreprocessUtil.preprocess(PreprocessUtil.tokenize(dialog[i + 1].strip()), max_len_emoji=params["max_len_emoji"])
                                if params["min_len_utterance"] <= len(s) <= params["max_len_utterance"]\
                                    and params["min_len_utterance"] <= len(t) <= params["max_len_utterance"]:
                                    nd += 1
                                    if nd <= params["max_num_data"]:
                                        if nd % 5000 == 0:
                                            print("written %d dialogs" % (nd))
                                        if params["multi_turn"] and len(context_toks) > 0:
                                            fs.write(" ".join(context_toks) + " ")

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
    outfile_source = "train_x_multi"
    outfile_target = "train_y_multi"

    params = {
        "max_num_data": 1000000,
        "min_len_utterance": 2,
        "max_len_utterance": 15,
        "max_len_emoji": 5,
        "multi_turn": True,
        "max_context_size": 5, # not including source
    }

    PairDataWriter.init()
    PairDataWriter.write_pairs(infile, outfile_source, outfile_target, params)
