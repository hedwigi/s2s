import os

params = {
    # Train Params
    "lr": 0.001,
    "keep_prob": 0.5,
    "batch_size": 128,
    "epochs": 50,
    "valid_step": 2000,
    "display_sample_per_n_batch": 300,
    "model_dir": "./model",
    "model_base": "s2s",
    "valid_size": 0.1,
    "n_samples2write": 10,
    "source_vocab": "data/train.source_vocab",
    "target_vocab": "data/train.target_vocab",

    # Model Params
    "pad_id": 0,  # padded with 0 in the model, not in file
    "start_id": 1,
    "end_id": 2,
    "unk_id": 3,  # since <UNK> should be created manually in DataLoader, should assign it with an id
    "source_vocab_size": 30000,
    "encoding_embedding_size": 128,
    "rnn_size": 256, # encoder decoder must have same number of layers and size
    "num_layers": 1,
    "target_vocab_size": 30000,
    "decoding_embedding_size": 128,

    "reverse_target": False,

    # TransformerKyu
    "maxlen": 16,  # Maximum number of words in a sentence. alias = T.
                 # Feel free to increase this if you are ambitious.
    "max_context_size": 6,
    "min_cnt": 5,  # words whose occurred less than min_cnt are encoded as <UNK>.
    "hidden_units": 512, # alias = C
    "num_blocks": 6,  # number of encoder/decoder blocks
    "num_epochs": 20,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "sinusoid": False,  # If True, use sinusoid. If false, positional embedding.
    "max_infer_step": 15,

    # Test Tiny model
    # "maxlen": 50,  # Maximum number of words in a sentence. alias = T.
    #              # Feel free to increase this if you are ambitious.
    # "max_context_size": 1,
    # "min_cnt": 2,  # words whose occurred less than min_cnt are encoded as <UNK>.
    # "hidden_units": 64, # alias = C
    # "num_blocks": 2,  # number of encoder/decoder blocks
    # "num_epochs": 20,
    # "num_heads": 2,
    # "dropout_rate": 0.1,
    # "sinusoid": False,  # If True, use sinusoid. If false, positional embedding.
    # "max_infer_step": 15,

}


LIB_DIR = "/Users/wangyuqian/PycharmProjects/libs"
LIB_PARAMS = {
    'jarpath': os.path.join(LIB_DIR, 'nlp-hanlp-1.2.12-RELEASE.jar'),
    'properties_path': LIB_DIR,
}
