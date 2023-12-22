import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torchvision.utils as vutils
import os


class BaseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): 路径到csv文件，包含标签。
            img_dir (string): 包含所有图像的目录。
            transform (callable, optional): 一个可选的transform来应用于样本。
        """
        self.face_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.face_labels)

    def __getitem__(self, idx):
        img_name = f'{self.img_dir}/{self.face_labels.iloc[idx, 0]}'
        image = Image.open(img_name)
        labels = self.get_labels(idx)
        sample = {'images': image, 'labels': labels}

        if self.transform:
            sample['images'] = self.transform(sample['images'])

        return sample

    def collate_fn(batch):
        images = [item['images'] for item in batch]
        labels = [item['labels'] for item in batch]

        images = torch.stack(images, dim=0).to(torch.float32)
        labels = torch.stack(labels, dim=0)

        return {'images': images, 'labels': labels}

    def get_labels(self, idx):
        raise NotImplementedError


class FaceDataset(BaseDataset):
    def __init__(self, csv_file, img_dir, transform=None):
        super().__init__(csv_file, img_dir, transform)

    def get_labels(self, idx):
        labels = self.face_labels.iloc[idx, 1:].to_numpy()
        labels = np.where(labels == -1, 0, labels)
        labels = torch.tensor(labels.astype('float32'))

        return labels


class HairDataset(BaseDataset):
    def __init__(self, csv_file, img_dir, transform=None):
        super().__init__(csv_file, img_dir, transform)

    def get_labels(self, idx):
        Hair_color = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
        labels_hair = self.face_labels.loc[idx, Hair_color].to_numpy()
        if all(label == -1 for label in labels_hair):
            labels_hair = np.append(labels_hair, 1)
        else:
            labels_hair = np.append(labels_hair, -1)
        labels_hair = np.where(labels_hair == -1, 0, labels_hair)
        labels_hair = labels_hair.astype('int')

        num_ones = np.sum(labels_hair == 1)
        if num_ones != 1:
            raise ValueError("labels_hair must contain exactly one '1'")

        label = np.where(labels_hair == 1)[0][0]
        label = torch.tensor(label).to(torch.long)

        return label


class SmileDataset(BaseDataset):
    def __init__(self, csv_file, img_dir, transform=None):
        super().__init__(csv_file, img_dir, transform)

    def get_labels(self, idx):
        labels_smile = self.face_labels.loc[idx, 'Smiling']
        if labels_smile == -1:
            labels_smile = 0
        labels_smile = torch.tensor(labels_smile).to(torch.long)

        return labels_smile


# csvfile = 'data_face_imgs/anno.csv'
# imgdir = 'data_face_imgs/images'
# FaceDataset = FaceDataset(csvfile, imgdir)
# sample = FaceDataset[2]
# labels = sample['labels']
# images = sample['images']
# vutils.save_image(images, 'test.png', normalize=True)
# print(labels)
