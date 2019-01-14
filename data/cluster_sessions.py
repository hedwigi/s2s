import os
import re
import time
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from util.PreprocessUtil import PreprocessUtil


def cluster_sessions(pathin, path_vocab, path_feat, path_clusters, path_centers,
                     ngram_size, stopwords_size, min_count, max_dialog_size,
                     n_clusters, random_state, threshold_tfidf):
    # get vocab, session reprz
    vocab = {}
    sessions = []
    dialog_ids = []

    start = time.clock()
    iter = dialog_iter(pathin)
    dialog = iter.__next__()
    id = 0
    while dialog:
        id += 1
        if id % 1000 == 0:
            print("processed %d dialogs" % id)

        words = []
        for line in dialog:
            # s = time.clock()
            tokens, postags = PreprocessUtil.tokenize_with_pos(line.strip())
            tokens, postags = PreprocessUtil.preprocess(tokens, postags)
            # print("t1 %f s" % (time.clock() - s))

            # s = time.clock()
            filtered_token_pos = []
            for j in range(len(tokens)):
                if not re.match(r'[' + PUNC + r'\s]*$', tokens[j]) \
                        and re.search(r'^[nvagijlst]', postags[j]):
                    filtered_token_pos.append(tokens[j] + "/" + postags[j])
            # print("t2 %f s" % (time.clock() - s))

            # ngrams (cross utterances)
            words += [w for w in PreprocessUtil.ngrams(filtered_token_pos, ngram_size)]

        # update vocab
        for w in words:
            vocab[w] = vocab.get(w, 0) + 1
        # update sessions
        if words:
            sessions.append(dict(Counter(words)))
            dialog_ids.append(id)

        if len(dialog_ids) == max_dialog_size:
            break

        dialog = iter.__next__()
    print("Constructed vocab and session representations for %d dialogs in %f s" % (id, time.clock() - start))

    start = time.clock()
    sorted_vocab = sorted(vocab.items(), key=lambda k_v: k_v[1], reverse=True)
    print("Sorted all vocab %f s" % (time.clock() - start))

    # write all vocab
    start = time.clock()
    with open(path_vocab, "w") as fout:
        for v, c in sorted_vocab:
            fout.write(v + " " + str(c) + "\n")
    print("Written out all vocab %f s" % (time.clock() - start))

    # remove stp
    start = time.clock()
    stopwords = set([w for w, _ in sorted_vocab[:stopwords_size]])
    for sess in sessions:
        # remove least freq words
        keys = list(sess.keys())
        for v in keys:
            if vocab[v] < min_count:
                del sess[v]

        # remove stp
        if stopwords:
            keys = list(sess.keys())
            for s in keys:
                if s in stopwords:
                    del sess[s]

    dialog_ids = [id for j, id in enumerate(dialog_ids) if sessions[j]]
    sessions = [sess for sess in sessions if sess]
    print("Removed stp and writing out all data %f s" % (time.clock() - start))

    # vectorize
    start = time.clock()
    vec = DictVectorizer()
    X = vec.fit_transform(sessions)
    print("DictVectorize %f s" % (time.clock() - start))

    start = time.clock()
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    X = tfidf_transformer.fit_transform(X)
    print("TfidfTransform %f s" % (time.clock() - start))

    start = time.clock()
    features = vec.get_feature_names()
    idfs = tfidf_transformer.idf_
    with open(path_feat, "w") as fout:
        for f in range(len(features)):
            fout.write("{0} {1:.2f}".format(features[f], idfs[f]) + "\n")
    print("Written features %f s" % (time.clock() - start))

    # clustering
    start = time.clock()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    pred = kmeans.labels_
    centers = kmeans.cluster_centers_
    print("Clustering %f s" % (time.clock() - start))

    # write session dict with label
    start = time.clock()
    with open(path_clusters, "w") as fout:
        for i in range(len(sessions)):
            fout.write(str(pred[i])
                       + "\t"
                       + str(dialog_ids[i])
                       + "\t"
                       + " ".join([w + ":" + str(c)
                                   for w, c in sorted(sessions[i].items(), key=lambda k_v:k_v[1], reverse=True)])
                       + "\n")
    print("Written clustering results %f s" % (time.clock() - start))

    # write centers
    start = time.clock()
    with open(path_centers, "w") as fout:
        for center in centers:
            fout.write(" ".join(["{0}:{1:.2f}".format(features[i], center[i])
                                 for i in range(len(center)) if center[i] > threshold_tfidf]) + "\n")
    print("Written centers %f s" % (time.clock() - start))


def dialog_iter(pathin):
    with open(pathin, "r") as fin:
        dialog = []
        line = fin.readline()
        while line:
            if not re.match(r'\s*$', line):
                dialog.append(line)
            else:
                yield dialog
                dialog = []

            line = fin.readline()

        if dialog:
            yield dialog

    yield None


def remove_headers(line):
    return re.sub(r'[UR]\[[0-9- :]+\]:', "", line.strip())


if __name__ == "__main__":

    redo_cluster = True
    dirdata = "/Users/wangyuqian/work/data"
    dirres = os.path.dirname(__file__)

    PUNC = open(os.path.join(os.path.dirname(__file__), "../resource/Punctuation"), "r").readline().strip()

    # Params
    pathin = os.path.join(dirdata, "multi_1_4.4_100w.data") if redo_cluster else None
    path_vocab = os.path.join(dirres, "clusters.vocab") if redo_cluster else None
    path_feat = os.path.join(dirres, "clusters.feat") if redo_cluster else None

    path_clusters = os.path.join(dirres, "clusters.res")
    path_centers = os.path.join(dirres, "clusters.centers")

    ngram_size = 2
    stopwords_size = 10
    min_count = 2
    n_clusters = 35
    random_state = 10
    max_dialog_size = 3000
    threshold_tfidf = 0.0

    PreprocessUtil.init()
    _ = PreprocessUtil.tokenize("你好呀")

    cluster_sessions(pathin, path_vocab, path_feat, path_clusters, path_centers,
                     ngram_size, stopwords_size, min_count, max_dialog_size,
                     n_clusters, random_state, threshold_tfidf)
