import torch



# %%
#导入数据集
from datasets import load_dataset

dataset = load_dataset(
    "IWSLT/iwslt2017",
    "iwslt2017-zh-en",
    trust_remote_code=True
)

# %%
train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

# 英文输入，中文输出
# train
train_en = [x["translation"]["en"] for x in train_data]
train_zh = [x["translation"]["zh"] for x in train_data]

# valid
valid_en = [x["translation"]["en"] for x in valid_data]
valid_zh = [x["translation"]["zh"] for x in valid_data]

# test
test_en = [x["translation"]["en"] for x in test_data]
test_zh = [x["translation"]["zh"] for x in test_data]

print("训练集数量：", len(train_en))
print("验证集数量：", len(valid_en))
print("测试集数量：", len(test_en))

# %%


all_data_en = train_en + valid_en + test_en
all_data_zh = train_zh + valid_zh + test_zh

chars_en = sorted(list(set(''.join(all_data_en))))
chars_zh = sorted(list(set(''.join(all_data_zh))))

print("英文词表数量：",len(chars_en))
print("中文词表数量：",len(chars_zh))
# %%



# _________________BPE_________________________________

from collections import Counter

def train_bpe(train_data, num_BPE, min_frequency):
    all_token = []
    for sentence in train_data:
        all_token.append(list(sentence))
    all_pair = []
    for step in range(num_BPE):
        statistics = Counter()
        for sentence in all_token:  #计算频次
            statistics.update(zip(sentence[:-1], sentence[1:]))

        if not statistics:
            break
        if statistics.most_common(1)[0][1] < min_frequency:
            break

        best_pair = statistics.most_common(1)[0][0]
        all_pair.append(best_pair)
        new_corpus = []
        for sentence in all_token:
            new_tokens = []
            j = 0
            while j < len(sentence):
                if sentence[j:j+2] == list(best_pair):
                    new_tokens.append(sentence[j] + sentence[j+1])
                    j += 2
                else:
                    new_tokens.append(sentence[j])
                    j += 1
            new_corpus.append(new_tokens)
        all_token = new_corpus
    return all_pair,all_token

# %%

# zh\en________________BPE_______________________

num_BPE_en = 295
num_BPE_zh = 333

min_frequency_en = 500
min_frequency_zh = 100

all_pair_en, final_tokens_en = train_bpe(
    train_en,
    num_BPE_en,
    min_frequency_en
)

all_pair_zh, final_tokens_zh = train_bpe(
    train_zh,
    num_BPE_zh,
    min_frequency_zh
)

# %%
# ______________________建立词典_________________________

def build_vocab(train_data, all_pair):
    chars = sorted(list(set(''.join(train_data))))
    for pair in all_pair:
        new_chars = pair[0] + pair[1]
        if new_chars not in chars:
            chars.append(new_chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}

    return chars, stoi, itos

# %%
chars_en, stoi_en, itos_en = build_vocab(train_en, all_pair_en)
chars_zh, stoi_zh, itos_zh = build_vocab(train_zh, all_pair_zh)

print("英文词表：", len(chars_en))
print("中文词表：", len(chars_zh))

print("英文 stoi：", len(stoi_en))
print("中文 stoi：", len(stoi_zh))

# %%

#___________________________encode___________________________

def encode(text, all_pair, stoi):
    token = list(text)

    for pair in all_pair:
        add_token = []
        j = 0
        while j < len(token):
            if token[j:j+2] == list(pair):
                add_token.append(token[j] + token[j+1])
                j += 2
            else:
                add_token.append(token[j])
                j += 1
        token = add_token
    
    # 遇到未知 Token，使用 <UNK> 代替
    token_id = [
        stoi.get(i, stoi["<UNK>"])
        for i in token
    ]

    return token_id
# %%
# decode
def decode(token_ids, itos):
    token = [itos[i] for i in token_ids]
    
    return ''.join(token)


# %%
# ________________________添加特殊token_________________________
chars_en, stoi_en, itos_en = build_vocab(train_en, all_pair_en)
if "<START>" not in chars_en:
    chars_en.extend(["<START>","<END>","<UNK>","<PAD>"])
stoi_en = {ch:i for i,ch in enumerate(chars_en)}
itos_en = {i:ch for i,ch in enumerate(chars_en)}


chars_zh, stoi_zh, itos_zh = build_vocab(train_zh, all_pair_zh)
if "<START>" not in chars_zh:
    chars_zh.extend(["<START>","<END>","<UNK>","<PAD>"])
stoi_zh = {ch:i for i,ch in enumerate(chars_zh)}
itos_zh = {i:ch for i,ch in enumerate(chars_zh)}
# %%

#________________________Padding______________________________________

def padding(batch, pad_id):
    max_len_sentence = max(len(sentence) for sentence in batch)
    padded_batch = []
    for sentence in batch:
        new_sentence = sentence.copy()
        while len(new_sentence) <max_len_sentence:
            new_sentence.append(pad_id)
        padded_batch.append(new_sentence)
    return padded_batch
# %%
print("stoi_en:",stoi_en["<PAD>"])
print("stoi_zh:",stoi_zh["<PAD>"])
# %%
#___________________________padding_mask_________________________

def create_padding_mask(batch, pad_id):
    mask = []
    for sentence in batch:
        sentence_mask = []
        for token in sentence:
            if token == pad_id:
                sentence_mask.append(0)
            else:
                sentence_mask.append(1)
        mask.append(sentence_mask)
    return mask
# %%

#___________________________get_batch_________________________
import random
def get_batch(train_en, train_zh, batch_size):
    batch_en = []
    batch_zh = []
    index = random.sample(range(len(train_en)), batch_size)
    for i in index:
        batch_en.append(train_en[i])
        batch_zh.append(train_zh[i])

    return batch_en, batch_zh
# %%
#___________________________batch_ID______________________________

def get_batch_ID(batch_en,batch_zh):
    batch_en_ID = []
    batch_zh_ID = []
    for sentence in batch_en:
        batch_en_ID.append(encode(sentence, all_pair_en, stoi_en))
        
    for   sentence in batch_zh:
        batch_zh_ID.append(encode(sentence, all_pair_zh, stoi_zh))

    return batch_en_ID,batch_zh_ID
# %%

def prepare_decoder_data(batch_zh_ID):
    start_zh_id = stoi_zh["<START>"]
    end_zh_id = stoi_zh["<END>"]

    decoder_input = []
    decoder_target = []

    for sentence in batch_zh_ID:

        new_input = sentence.copy()
        new_input.insert(0, start_zh_id)

        new_target = sentence.copy()
        new_target.append(end_zh_id)

        decoder_input.append(new_input)
        decoder_target.append(new_target)

    return decoder_input, decoder_target


# %%

#_________________________get_batch_data整合________________________________

def get_batch_data(train_en, train_zh, batch_size):

    batch_en, batch_zh = get_batch(train_en, train_zh, batch_size)
    
    batch_en_ID,batch_zh_ID = get_batch_ID(batch_en,batch_zh)

    decoder_input, decoder_target = prepare_decoder_data(batch_zh_ID)


    pad_en = stoi_en["<PAD>"]
    pad_zh = stoi_zh["<PAD>"]
    
    
    padded_en_batch = padding(batch_en_ID, pad_en)
    padded_decoder_input = padding(decoder_input, pad_zh)
    padded_decoder_target = padding(decoder_target, pad_zh)

    encoder_mask = create_padding_mask(padded_en_batch, pad_en)
    decoder_mask = create_padding_mask(padded_decoder_input, pad_zh)
    target_mask = create_padding_mask(padded_decoder_target, pad_zh)
    
    
    padded_en_batch_tensor = torch.tensor(padded_en_batch, dtype=torch.long)
    padded_decoder_input_tensor = torch.tensor(padded_decoder_input, dtype=torch.long)
    padded_decoder_target_tensor = torch.tensor(padded_decoder_target, dtype=torch.long)
    encoder_mask_tensor = torch.tensor(encoder_mask, dtype=torch.bool)
    decoder_mask_tensor = torch.tensor(decoder_mask, dtype=torch.bool)
    target_mask_tensor = torch.tensor(target_mask, dtype=torch.bool)
    


    return padded_en_batch_tensor,padded_decoder_input_tensor,padded_decoder_target_tensor,encoder_mask_tensor,decoder_mask_tensor,target_mask_tensor

