import torch
import torch.nn as nn

from dataset import get_batch_data, train_en, train_zh, valid_en, valid_zh
from model_setup import batch_size, device, model, pad_id_zh, vocab_size_zh


def train():
    learning_rate = 1e-3

    #最佳模型
    best_val_loss = float("inf")

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id_zh)

    #  AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # %%
    #____________________________________训练_________________________________

    max_steps = 5000
    model.train()

    #_________________________________评估验证集______________________________
    def estimate_loss(model, valid_en, valid_zh, batch_size):

        model.eval()
        losses = []
        with torch.no_grad():

            for i in range(20):

                (
                    encoder_input,
                    decoder_input,
                    decoder_target,
                    encoder_mask,
                    decoder_mask,
                    target_mask
                ) = get_batch_data(valid_en, valid_zh, batch_size)

                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                decoder_target = decoder_target.to(device)
                encoder_mask = encoder_mask.to(device)
                decoder_mask = decoder_mask.to(device)

                # 前向传播
                logits = model(
                    encoder_input,
                    decoder_input,
                    encoder_mask=encoder_mask,
                    decoder_mask=decoder_mask
                )

                # 计算验证 Loss
                logits_flat = logits.reshape(-1, vocab_size_zh)
                target_flat = decoder_target.reshape(-1)
                loss = criterion(logits_flat, target_flat)
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        model.train()

        return avg_loss

    #保存最佳模型

    for step in range(max_steps):
        (
            encoder_input,
            decoder_input,
            decoder_target,
            encoder_mask,
            decoder_mask,
            target_mask
        ) = get_batch_data(train_en, train_zh, batch_size)

        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)
        encoder_mask = encoder_mask.to(device)
        decoder_mask = decoder_mask.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(
        encoder_input,
        decoder_input,
        encoder_mask=encoder_mask,
        decoder_mask=decoder_mask
        )

        #计算 Loss
        logits_flat = logits.reshape(-1, vocab_size_zh)
        target_flat = decoder_target.reshape(-1)
        loss = criterion(logits_flat, target_flat)

        #反向传播与参数更新
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
            )

        optimizer.step()


        if step % 100 == 0 or step == max_steps - 1:

            val_loss = estimate_loss(
                model,
                valid_en,
                valid_zh,
                batch_size
                )

            print(
                f"Step {step} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_transformer.pth")
                print("已保存最佳模型")

    return optimizer


if __name__ == "__main__":
    train()
