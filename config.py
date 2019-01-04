import os

params = {
    # Train Params
    "lr": 0.0001,
    "keep_prob": 0.8,
    "batch_size": 32,
    "epochs": 10,
    "valid_step": 20,
    "display_sample_per_n_batch": 5,
    "model_dir": "./model",
    "model_base": "s2s",
    "valid_size": 0.2,

    # Model Params
    "pad_id": 0,  # padded with 0 in the model, not in file
    "start_id": 1,
    "end_id": 2,
    "unk_id": 3,  # since <UNK> should be created manually in DataLoader, should assign it with an id
    "source_vocab_size": 30000,
    "encoding_embedding_size": 128,
    "rnn_size": 256, # encoder decoder must have same number of layers and size
    "num_layers": 2,
    "target_vocab_size": 30000,
    "decoding_embedding_size": 128,

}


LIB_DIR = "/Users/wangyuqian/PycharmProjects/libs"
LIB_PARAMS = {
    'jarpath': os.path.join(LIB_DIR, 'nlp-hanlp-1.2.12-RELEASE.jar'),
    'properties_path': LIB_DIR,
}
