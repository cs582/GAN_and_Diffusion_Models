import os
import torch


def save_model(model, name, epoch, saving_path):
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    filename = f"{saving_path}/{name}_epoch{epoch}.pt"

    torch.save(model.state_dict(), filename)
