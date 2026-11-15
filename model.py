import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()

        position = torch.arange(max_len)
        i = torch.arange(0, embedding_dim, 2).float()
        div_term = 10000 ** (i / embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)

        angle = position.unsqueeze(1) / div_term
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        
        self.register_buffer("pe", pe)   #不参与梯度训练

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return x
        
# %%

#___________________________MultiHeadAttention_______________________________

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads


        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.out_MHA = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x, padding_mask=None, causal=False):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        #Q、K、V 拆分成多个 Head
        Q = Q.reshape(B, T, self.num_heads, self.head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim)
        V = V.reshape(B, T, self.num_heads, self.head_dim)
        #交换维度
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        #计算得分
        scores = Q @ K.transpose(-2, -1)
        scores = scores * (Q.shape[-1] ** -0.5)

        # Causal Mask
        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Padding Mask
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))


        # Softmax，计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = attention_weights @ V

        # 拼接 Head
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(B, T, C)

        # 输出线性投影
        output = self.out_MHA(attention_output)

        return output


# %%

#___________________________________EncoderBlock_________________________________
class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super().__init__()

        # 创建 MultiHeadAttention
        self.attention = MultiHeadAttention(embedding_dim,num_heads)

        # 创建第一个 LayerNorm
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 创建 FFN 的第一层 Linear
        self.fc1 = nn.Linear(embedding_dim, ffn_dim)

        # 创建 ReLU
        self.relu = nn.ReLU()

        # 创建 FFN 的第二层 Linear
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)

        # 创建第二个 LayerNorm
        self.norm2 = nn.LayerNorm(embedding_dim)



    def forward(self, x, padding_mask=None):

        # Multi-Head Self-Attention
        attention_output = self.attention(x,padding_mask=padding_mask,causal=False)

        # 第一次残差连接 + LayerNorm
        y = self.norm1(x + attention_output)

        # Feed Forward Network
        ffn_output = self.fc1(y)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.fc2(ffn_output)

        # 第二次残差连接 + LayerNorm
        z = self.norm2(y + ffn_output)

        return z
# %%
#___________________________________Encoder______________________________________
class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, num_layers):
        super().__init__()

        # 创建多个 EncoderBlock
        self.layers = nn.ModuleList([
                      EncoderBlock(embedding_dim, num_heads, ffn_dim)
                      for _ in range(num_layers)
])

    def forward(self, x, padding_mask=None):

        # 依次经过每个 EncoderBlock
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        return x


# %%

#_____________________________CrossAttention_________________________________

class CrossAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.head_dim = embedding_dim // num_heads
        
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out_MHA = nn.Linear(embedding_dim, embedding_dim)
        

    def forward(self, x, context, padding_mask=None):

        B_zh, T_zh, C_zh = x.shape
        B_en, T_en, C_en = context.shape

        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)
        # 拆分多个 Head
        Q = Q.reshape(B_zh, T_zh, self.num_heads, self.head_dim)
        K = K.reshape(B_en, T_en, self.num_heads, self.head_dim)
        V = V.reshape(B_en, T_en, self.num_heads, self.head_dim)
        #交换维度
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        #计算得分
        scores = Q @ K.transpose(-2, -1)
        scores = scores * (Q.shape[-1] ** -0.5)
        # Padding Mask
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))
        # Softmax，计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = attention_weights @ V

        # 拼接 Head
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(B_zh, T_zh, C_zh)

        # 输出线性投影
        output = self.out_MHA(attention_output)

        return output

# %%
#_______________________________DecoderBlock______________________________

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super().__init__()

        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(embedding_dim,num_heads)

        # 第一个 LayerNorm
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Cross-Attention
        self.cross_attention = CrossAttention(embedding_dim, num_heads)

        # 第二个 LayerNorm
        self.norm2 = nn.LayerNorm(embedding_dim)

        # FFN
        self.fc1 = nn.Linear(embedding_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)

        # 第三个 LayerNorm
        self.norm3 = nn.LayerNorm(embedding_dim)


    def forward(self, x, context, decoder_mask=None, encoder_mask=None):
        
        # Multi-Head Self-Attention
        self_attention_output = self.self_attention(x,padding_mask=decoder_mask,causal=True)

        # 第一次残差连接 + LayerNorm
        y = self.norm1(x + self_attention_output)

        # Cross-Attention
        cross_attention_output = self.cross_attention(y,context,padding_mask=encoder_mask)

        # 第二次残差连接 + LayerNorm
        z = self.norm2(y + cross_attention_output)

        # FFN
        ffn_output = self.fc1(z)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.fc2(ffn_output)

        # 第三次残差连接 + LayerNorm
        output = self.norm3(z + ffn_output)

        return output


# %%
#___________________________________Decoder________________________________________



class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, num_layers):
        super().__init__()

        # 创建多个 DecoderBlock
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
            ])

    def forward(self, x, context, decoder_mask=None, encoder_mask=None):

        # 让 x 依次经过每一个 DecoderBlock
        for layer in self.layers:
            x = layer(x, context, decoder_mask, encoder_mask)

        return x


# %%

#_____________________________________Transformer_________________________________

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size_en,
        vocab_size_zh,
        embedding_dim,
        num_heads,
        ffn_dim,
        num_layers,
        pad_id_en,
        pad_id_zh
    ):
        super().__init__()

        # 英文 Embedding
        self.en_embedding = nn.Embedding(vocab_size_en, embedding_dim, padding_idx=pad_id_en)

        # 中文 Embedding
        self.zh_embedding = nn.Embedding(vocab_size_zh, embedding_dim, padding_idx=pad_id_zh)

        # 位置编码
        self.pos_encoder = PositionalEncoding(embedding_dim,max_len=512)

        # Encoder
        self.encoder = Encoder(embedding_dim, num_heads, ffn_dim, num_layers)

        # Decoder
        self.decoder = Decoder(embedding_dim, num_heads, ffn_dim, num_layers)

        # 中文词表输出 Linear
        self.output_linear = nn.Linear(embedding_dim, vocab_size_zh)


    def forward(
    self,
    encoder_input,
    decoder_input,
    encoder_mask=None,
    decoder_mask=None
    ):

        # 英文 Embedding + PE
        en_emb = self.en_embedding(encoder_input)
        en_emb = self.pos_encoder(en_emb)

        # 中文 Embedding + PE
        zh_emb = self.zh_embedding(decoder_input)
        zh_emb = self.pos_encoder(zh_emb)

        # Encoder
        context = self.encoder(en_emb, padding_mask=encoder_mask)

        # Decoder
        decoder_output = self.decoder(zh_emb,context,decoder_mask=decoder_mask,encoder_mask=encoder_mask)

        # 输出中文词表 Logits
        logits = self.output_linear(decoder_output)

        return logits

# %%

