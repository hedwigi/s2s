import os

from collections import defaultdict


# Common Params
params_common = {
    # *** data params
    "datadir": "data/en-fr",
    "data_suffix": "small",

    # *** vocab params
    "pad_id": 0,  # padded with 0 in the model, not in file
    "start_id": 1,
    "end_id": 2,
    "unk_id": 3,  # since <UNK> should be created manually in DataLoader, should assign it with an id
    "source_vocab_size": 30000,
    "target_vocab_size": 30000,
    "source_vocab": "data/train.source_vocab",
    "target_vocab": "data/train.target_vocab",
    "reverse_target": False,

    # *** training params
    "mode": "train",                        # "single"|"valid_batch"
    "model_name": ["Transformer", "Test"],  # ["S2S"]
    "batch_size": 128,
    "epochs": 50,
    "model_dir": "./model",
    "model_base": "s2s",

    # *** valid params
    "valid_step": 200,
    "display_sample_per_n_batch": 50,
    "n_samples2write": 10,
    "results_dir": "results/",
    "results_base": "trfm",
}

BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=params_common["batch_size"],  # Maximum number of tokens per batch of examples.

    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=params_common["source_vocab_size"],  # Number of tokens defined in the vocabulary file.
    hidden_size=512,  # Model dimension in the hidden layers.
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_heads=8,  # Number of heads to use in multi-headed attention.
    filter_size=2048,  # Inner layer dimension in the feedforward network.

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=5,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization in beam search

    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,
)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    hidden_size=1024,
    filter_size=4096,
    num_heads=16,
),

TEST_PARAMS = BASE_PARAMS.copy()
TEST_PARAMS.update(
    hidden_size=32,
    filter_size=64,
    num_hidden_layers=2,
    num_heads=2,
)

params_models = {

    "S2S": {
        # Train Params
        "lr": 0.001,
        "keep_prob": 0.5,
        "encoding_embedding_size": 200,
        "rnn_size": 128, # encoder decoder must have same number of layers and size
        "num_layers": 1,
        "decoding_embedding_size": 200,
        },

    "Transformer": {
        "Base": BASE_PARAMS,
        "Big": BIG_PARAMS,
        "Test": TEST_PARAMS,
    },

}


LIB_DIR = "/Users/wangyuqian/PycharmProjects/libs"
LIB_PARAMS = {
    'jarpath': os.path.join(LIB_DIR, 'nlp-hanlp-1.2.12-RELEASE.jar'),
    'properties_path': LIB_DIR,
}
