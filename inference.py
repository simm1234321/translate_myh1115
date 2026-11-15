import torch

from dataset import all_pair_en, encode, itos_zh, stoi_en, stoi_zh
from model_setup import device, model, pad_id_en

# %%
#____________________________________translate_________________________________

def translate(sentence, max_new_tokens=100):
    
    model.eval()

    token_ID = encode(sentence, all_pair_en, stoi_en)


    encoder_input = torch.tensor(
    [token_ID],
    dtype=torch.long,
    device=device
    )

    # encoder_mask
    encoder_mask = encoder_input != pad_id_en

    # decoder_input
    decoder_input = torch.tensor(
    [[stoi_zh["<START>"]]],
    dtype=torch.long,
    device=device
    )

# 自回归生成
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(
                    encoder_input,
                    decoder_input,
                    encoder_mask=encoder_mask,
                    )
            
            next_token_logits = logits[:, -1, :]
            
            # next_token = torch.argmax(
            #     next_token_logits,
            #     dim=-1,
            #     keepdim=True
            #     )
            prob = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)

            if next_token.item() != stoi_zh["<END>"]:
                #拼接
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
            else:
                break
        
        decoder_input = decoder_input[0, 1:]
        ids = decoder_input.tolist()
        tokens = [itos_zh[i] for i in ids]
        output = "".join(tokens)


        return output



def load_checkpoint():
    # %%
    state_dict = torch.load(
        "best_transformer.pth",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()


def run_examples():
    # %%
    sentences = [

        "How are you.",
        "I love you.",
        "Hello.",
        "Thank you.",
        "I am a student.",
        "What is your name?"
    ]

    for sentence in sentences:
        print("英文：", sentence)
        print("中文：", translate(sentence))
        print()


if __name__ == "__main__":
    load_checkpoint()
    run_examples()
