import torch
import os
import random
import numpy as np


def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),  # 保存调度器状态
        'epoch': epoch
    }, path)


def load_model(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch']
    return start_epoch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_sigmoid(outputs):
    outputs = outputs.cpu().numpy()
    positive_mask = outputs >= 0
    negative_mask = ~positive_mask
    z = np.empty_like(outputs)

    z[positive_mask] = 1 / (1 + np.exp(-outputs[positive_mask]))

    z[negative_mask] = np.exp(outputs[negative_mask]) / \
        (1 + np.exp(outputs[negative_mask]))

    return z
