# train.py

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import resnet18, resnet34, resnet50, resnet101, resnet152
from datasets import FaceDataset, HairDataset, SmileDataset
from trainer import Trainer
import numpy as np
import random
import wandb
import datetime
import os
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a deep learning model.")
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Training batch size')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--config', type=str,
                        help='Path to the config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the model to resume training')
    parser.add_argument('--data_dir', type=str,
                        default='data', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str,
                        default='output', help='Path to the output directory')
    parser.add_argument('--task', type=str, default="all",
                        help='all or hair or smile')
    parser.add_argument('--wandb', default=True, help='use wandb')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--model', type=str, default="resnet34",
                        help='model name')
    parser.add_argument('--warmup', type=int, default=4,
                        help='warmup epochs')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    if args.config:
        config = load_config(args.config)
        args.epochs = config.get('epochs', args.epochs)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.learning_rate = config.get('learning_rate', args.learning_rate)
        args.resume = config.get('resume', args.resume)
        args.data_dir = config.get('data_dir', args.data_dir)
        args.output_dir = config.get('output_dir', args.output_dir)
        args.task = config.get('task', args.task)
        args.wandb = config.get('wandb', args.wandb)
        args.seed = config.get('seed', args.seed)
        args.model = config.get('model', args.model)
        args.marmup = config.get('warmup', args.warmup)
        args.pretrained = config.get('pretrained', args.pretrained)

    seed_everything(seed=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = args.data_dir
    train_data_dir = f'{data_dir}/image_train'
    val_data_dir = f'{data_dir}/image_test'
    csv_train = f'{data_dir}/train_anno.csv'
    csv_val = f'{data_dir}/test_anno.csv'

    run_time = "run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(args.output_dir, args.model)
    args.output_dir = os.path.join(args.output_dir, run_time)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task == "all":
        dataset_train = FaceDataset(csv_train, train_data_dir)
        dataset_val = FaceDataset(csv_val, val_data_dir)
    elif args.task == "hair":
        dataset_train = HairDataset(csv_train, train_data_dir)
        dataset_val = HairDataset(csv_val, val_data_dir)
    elif args.task == "smile":
        dataset_train = SmileDataset(csv_train, train_data_dir)
        dataset_val = SmileDataset(csv_val, val_data_dir)
    else:
        raise ValueError("task must be all or hair")

    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=FaceDataset.collate_fn, num_workers=4)
    val_loader = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=FaceDataset.collate_fn, num_workers=4)

    if args.task == "all":
        args.num_classes = 40
        criterion = nn.BCEWithLogitsLoss()
    elif args.task == "hair":
        args.num_classes = 5
        criterion = nn.CrossEntropyLoss()
    elif args.task == "smile":
        args.num_classes = 2
        criterion = nn.CrossEntropyLoss()

    model_name = args.model

    if args.pretrained:
        if model_name in globals():
            model = globals()[model_name](
                pretrained=args.pretrained)
        else:
            raise ValueError(
                "Model name must be one of: resnet18, resnet34, resnet50, resnet101, resnet152")
        model.fc = nn.Linear(model.fc.in_features,
                             args.num_classes)
    else:
        if model_name in globals():
            model = globals()[model_name](n_classes=args.num_classes)
        else:
            raise ValueError(
                "Model name must be one of: resnet18, resnet34, resnet50, resnet101, resnet152")

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as file:
        json.dump(vars(args), file)

    if args.wandb:
        wandb.init(project="face-classification", config=args)
        wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler1 = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, total_iters=(args.warmup-1)*len(train_loader))
    scheduler2 = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.1)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[(args.warmup-1)*len(train_loader)])

    trainer = Trainer(model, train_loader, val_loader,
                      optimizer, scheduler, criterion, device, args)

    trainer.train()

    wandb.finish()


if __name__ == '__main__':
    main()
