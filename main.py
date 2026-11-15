import torch

from inference import load_checkpoint, run_examples
from model_setup import model
from train import train


if __name__ == "__main__":
    optimizer = train()
    load_checkpoint()
    run_examples()
    # %%
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, "transformer_collapsed_backup.pth")
