# Configuration for training
training_config = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_steps": 400000
}

# Configuration for dataset
dataset_config = {
    "l_noise": 100,  # number of padding tokens
    "l_memorize": 16,  # number of tokens to memorize
    "n_tokens": 16,  # alphabet size
    "lag": False,
    "variable": True,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
    "variable_length": False,  # Randomize number of tokens to memorize
    "one_hot": False,
    "reverse": False,
    "static": False,
}

# Configuration for Mamba model
class MambaConfig:
    d_model: int = 64
    d_intermediate: int = 0
    n_layer: int = 2
    vocab_size: int = dataset_config['n_tokens']
    ssm_cfg: dict = {}
    attn_layer_idx: list = []
    attn_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False

# class MambaConfig:

#     d_model: int = 2560
#     d_intermediate: int = 0
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = dict
#     attn_layer_idx: list = list
#     attn_cfg: dict = dict
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8
#     tie_embeddings: bool = True