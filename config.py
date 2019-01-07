import os

params = {
    # Train Params
    "lr": 0.001,
    "keep_prob": 0.5,
    "batch_size": 128,
    "epochs": 50,
    "valid_step": 200,
    "display_sample_per_n_batch": 50,
    "model_dir": "./model",
    "model_base": "s2s",
    "valid_size": 0.2,
    "n_samples2write": 10,

    # Model Params
    "pad_id": 0,  # padded with 0 in the model, not in file
    "start_id": 1,
    "end_id": 2,
    "unk_id": 3,  # since <UNK> should be created manually in DataLoader, should assign it with an id
    "source_vocab_size":8000,
    "encoding_embedding_size": 100,
    "rnn_size": 128, # encoder decoder must have same number of layers and size
    "num_layers": 3,
    "target_vocab_size": 8000,
    "decoding_embedding_size": 100,

}


LIB_DIR = "/Users/wangyuqian/PycharmProjects/libs"
LIB_PARAMS = {
    'jarpath': os.path.join(LIB_DIR, 'nlp-hanlp-1.2.12-RELEASE.jar'),
    'properties_path': LIB_DIR,
}
