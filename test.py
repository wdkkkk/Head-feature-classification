import argparse
import torch
from torch.utils.data import DataLoader
from models import resnet18, resnet34, resnet50, resnet101, resnet152
from datasets import FaceDataset, HairDataset, SmileDataset
import os
import json
from utils import stable_sigmoid
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import confusion_matrix
from utils import seed_everything
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing a deep learning model.")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Testing batch size')
    parser.add_argument('--task', type=str, default="all",
                        help='all or hair or smile')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def load_model(model_name, checkpoint_path, num_classes):
    if model_name in globals():
        model = globals()[model_name]()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(
            "Model name must be one of: resnet18, resnet34, resnet50, resnet101, resnet152")

    checkpoint = torch.load(checkpoint_path)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    return model


def analyze_all(outputs, labels):
    outputs_logistic = stable_sigmoid(outputs.detach())
    correct_num_class = np.sum((outputs_logistic > 0.5)
                               == labels.cpu().numpy(), axis=0)
    acc_class = correct_num_class / labels.size(0)

    plt.figure(figsize=(10, 7))
    plt.bar(range(40), acc_class)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per class')
    plt.savefig('nn_acc_per_class.png', dpi=300)


def analyze_hair(outputs, labels):
    outputs_logistic = torch.softmax(outputs.detach(), dim=1)
    predicted = torch.argmax(outputs_logistic, dim=1)
    acc = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
    precision = precision_score(
        labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    recall = recall_score(
        labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    f1 = f1_score(labels.cpu().numpy(),
                  predicted.cpu().numpy(), average='macro')
    print(f"acc: {acc}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}")
    cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('nn_confusion_matrix2.png', dpi=300)

    print(classification_report(
        labels.cpu().numpy(), predicted.cpu().numpy()))


def analyze_smile(outputs, labels):
    outputs_logistic = torch.softmax(outputs.detach(), dim=1)
    outputs_logistic_one = outputs_logistic[:, 1]
    predicted = torch.argmax(outputs_logistic, dim=1)
    correct_num = (predicted == labels).sum().item()
    acc = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
    precision = precision_score(
        labels.cpu().numpy(), predicted.cpu().numpy())
    recall = recall_score(
        labels.cpu().numpy(), predicted.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())
    roc_auc = roc_auc_score(labels.cpu().numpy(),
                            outputs_logistic_one.cpu().numpy())
    print(
        f"acc: {acc}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nroc_auc: {roc_auc}")
    cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('nn_confusion_matrix.png', dpi=300)

    roc = roc_curve(labels.cpu().numpy(), outputs_logistic_one.cpu().numpy())
    plt.figure(figsize=(10, 7))
    plt.plot(roc[0], roc[1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('nn_roc_curve.png', dpi=300)


def main():
    args = parse_args()
    if args.config:
        config = load_config(args.config)
        args.model = config.get('model', args.model)
        args.checkpoint = config.get('checkpoint', args.checkpoint)
        args.data_dir = config.get('data_dir', args.data_dir)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.task = config.get('task', args.task)
        args.seed = config.get('seed', args.seed)

    seed_everything(seed=args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define num_classes based on task
    if args.task == "all":
        num_classes = 40
    elif args.task == "hair":
        num_classes = 5
    elif args.task == "smile":
        num_classes = 2
    else:
        raise ValueError("Task must be all, hair, or smile")

    model = load_model(args.model, args.checkpoint, num_classes)
    model = model.to(device)

    test_data_dir = f'{args.data_dir}/image_test'
    csv_test = f'{args.data_dir}/test_anno.csv'

    if args.task == "all":
        dataset_test = FaceDataset(csv_test, test_data_dir)
    elif args.task == "hair":
        dataset_test = HairDataset(csv_test, test_data_dir)
    elif args.task == "smile":
        dataset_test = SmileDataset(csv_test, test_data_dir)

    test_loader = DataLoader(dataset_test, batch_size=args.batch_size,
                             shuffle=False, collate_fn=FaceDataset.collate_fn, num_workers=4)

    model.eval()
    with torch.no_grad():
        outputs_list = []
        labels_list = []
        for data in test_loader:
            images, labels = data['images'], data['labels']
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            labels_list.append(labels)
            outputs_list.append(outputs)

        outputs = torch.cat(outputs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

    if args.task == "all":
        analyze_all(outputs, labels)
    elif args.task == "hair":
        analyze_hair(outputs, labels)
    elif args.task == "smile":
        analyze_smile(outputs, labels)
    print('Testing completed')


if __name__ == '__main__':
    main()
