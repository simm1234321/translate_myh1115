import torch
import torch.nn as nn

from dataset import chars_en, chars_zh, stoi_en, stoi_zh
from model import Transformer

# %%
#__________________________________Embedding_________________________________

en_embedding = nn.Embedding(
    num_embeddings=len(chars_en),
    embedding_dim=256,
    padding_idx=stoi_en["<PAD>"]
)

zh_embedding = nn.Embedding(
    num_embeddings=len(chars_zh),
    embedding_dim=256,
    padding_idx=stoi_zh["<PAD>"]
)


# ________________________________________模型初始化________________________________________

embedding_dim = 256
num_heads = 8
ffn_dim = 1024
num_layers = 2
batch_size = 4

pad_id_en = stoi_en["<PAD>"]
pad_id_zh = stoi_zh["<PAD>"]

vocab_size_en = len(chars_en)
vocab_size_zh = len(chars_zh)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# 创建 Transformer
model = Transformer(
    vocab_size_en = vocab_size_en,
    vocab_size_zh = vocab_size_zh,
    embedding_dim = embedding_dim,
    num_heads = num_heads,
    ffn_dim = ffn_dim,
    num_layers = num_layers,
    pad_id_en = pad_id_en,
    pad_id_zh = pad_id_zh,
)

# 将模型移动到设备
model = model.to(device)
