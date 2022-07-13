"""
Author: Greg Holste
Last Modified: 12/9/21
Description: PyTorch Dataset object for loading and preprocessing NIH ChestXRay14 dataset (images + metadata).
"""

import os

import cv2
import numpy as np
import pandas as pd

import torch
import torchvision
from sklearn.utils import compute_class_weight

class ChestXRay14(torch.utils.data.Dataset):
    """PyTorch Dataset for NIH ChestXRay14 dataset."""
    
    def __init__(self, data_dir, split, augment, n_TTA=0, n=-1):
        self.img_dir = os.path.join(data_dir, 'images')
        self.split = split
        self.augment = augment
        self.n_TTA = n_TTA
        self.n = n

        self.CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

        self.label_df = pd.read_csv(f'{self.split}_list.txt', delim_whitespace=True, names=['file'] + self.CLASSES)
        self.label_df = self.label_df.sort_values(by='file')

        self.files = self.label_df['file'].values.tolist()
        self.labels = self.label_df[self.CLASSES].values

        self.pos_weight = compute_class_weight(class_weight='balanced', classes=list(range(8)), y=self.labels.argmax(axis=1))

        # Define augmentation pipeline for training and testing (when TTA enabled)
        if self.augment or self.n_TTA > 0:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(degrees=5),
                torchvision.transforms.ColorJitter(contrast=0.25),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
        # For validation and testing (with TTA disabled), simply resize to 224x224
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])


    # def __init__(self, data_dir, split, augment, n_TTA=0, n=-1):
    #     self.img_dir = os.path.join(data_dir, 'images')
    #     self.metadata_path = os.path.join(data_dir, 'DataEntry_2017_v2020.csv')
    #     self.split = split
    #     self.augment = augment
    #     self.n_TTA = n_TTA
    #     self.n = n

    #     self.metadata_df = pd.read_csv(self.metadata_path)

    #     self.CLASSES = sorted(list(set([s.split('|')[0] for s in self.metadata_df["Finding Labels"] if len(s.split('|')) == 1])))

    #     if self.split == 'train' or self.split == 'val':
    #         self.files = sorted(list(np.unique(pd.read_csv(os.path.join(data_dir, f'{self.split}_only_15.txt'), delim_whitespace=True).iloc[:, 0].values)))
    #     else:
    #         self.files = sorted(list(np.unique(pd.read_csv(os.path.join(data_dir, f'{self.split}_final_15.txt'), delim_whitespace=True).iloc[:, 0].values)))

    #     if self.n != -1:
    #         self.files = self.files[:n]

    #     self.metadata_df = self.metadata_df[self.metadata_df['Image Index'].isin(self.files)]
    #     self.metadata = self.prepare_metadata(self.metadata_df)
    #     self.meta_features = self.metadata.shape[1]
    #     self.labels = np.stack(self.metadata_df.apply(lambda x: self.to_one_hot(x["Finding Labels"]), axis=1))

    #     if self.split == 'train':
    #         # self.pos_weight = compute_class_weight('balanced', list(range(15)), self.labels.argmax(axis=1))
    #         # self.pos_weight = self.pos_weight / np.min(self.pos_weight)

    #         class_freqs = self.labels.sum(0)
    #         self.pos_weight = np.array([class_freqs[self.CLASSES.index('No Finding')] / x for x in class_freqs])


    #     self.labels = np.delete(self.labels, self.CLASSES.index('No Finding'), 1)
    #     if self.split == 'train':
    #         self.pos_weight = self.pos_weight[np.where(np.array(self.CLASSES) != 'No Finding')[0]]
    #     self.CLASSES = list(np.array(self.CLASSES)[np.where(np.array(self.CLASSES) != 'No Finding')[0]])


    #     # Define augmentation pipeline for training and testing (when TTA enabled)
    #     if self.augment or self.n_TTA > 0:
    #         self.transform = torchvision.transforms.Compose([
    #             torchvision.transforms.ToPILImage(),
    #             torchvision.transforms.RandomCrop(224),
    #             torchvision.transforms.RandomHorizontalFlip(),
    #             torchvision.transforms.RandomRotation(degrees=5),
    #             torchvision.transforms.ColorJitter(contrast=0.25),
    #             torchvision.transforms.ToTensor(),
    #             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                              std=[0.229, 0.224, 0.225])
    #         ])
    #     # For validation and testing (with TTA disabled), simply resize to 224x224
    #     else:
    #         self.transform = torchvision.transforms.Compose([
    #             torchvision.transforms.ToPILImage(),
    #             torchvision.transforms.ToTensor(),
    #             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                              std=[0.229, 0.224, 0.225])
    #         ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = cv2.imread(os.path.join(self.img_dir, self.files[idx]), cv2.IMREAD_GRAYSCALE)

        # For training and testing (with TTA), resize to 256x256 and augment
        if self.augment or self.n_TTA > 0:
            x = cv2.resize(x, (256, 256))
        # Otherwise, for validation and testing (w/o TTA), resize to 224x224
        else:
            x = cv2.resize(x, (224, 224))

        # Repeat across color channels        
        x = np.stack([x, x, x], axis=-1)

        # Perform test-time augmentation (get n_TTA augmented copies of image and stack along first dim)
        if self.n_TTA > 0:
            x = torch.stack([self.transform(x) for _ in range(self.n_TTA)], dim=-1)
        # Perform training augmentation
        else:
            x = self.transform(x)

        y = self.labels[idx]

        return {'x': x.float(), 'y': torch.from_numpy(y).float()}

        # meta = self.metadata[idx]
        # y = self.labels[idx]

        # return {'x': x.float(), 'y': torch.from_numpy(y).float(), 'meta': torch.from_numpy(meta).float()}

    # def prepare_metadata(self, x):
    #     # Mean metadata feature values obtained from training set
    #     MEAN_AGE = 46.609333
    #     STD_AGE = 16.684651
    #     # MEAN_IMG_WIDTH = 2641.169426
    #     # STD_IMG_WIDTH = 336.751674
    #     # MEAN_IMG_HEIGHT = 2504.875279
    #     # STD_IMG_HEIGHT = 404.008517
    #     MEAN_PX_SPACING = 0.155413
    #     STD_PX_SPACING = 0.015608

    #     # Extract age, sex, view (AP or PA), and pixel spacing. Normalize age and pixel spacing and one-hot-encode sex and view (dropping redundant level).
    #     age = ((x['Patient Age'] - MEAN_AGE) / STD_AGE).values[:, None]
    #     sex = pd.get_dummies(x['Patient Gender'], drop_first=True).values
    #     view = pd.get_dummies(x['View Position'], drop_first=True).values
    #     pixel_spacing = ((x['OriginalImagePixelSpacing[x'] - MEAN_PX_SPACING) / STD_PX_SPACING)[:, None]

    #     return np.concatenate([age, sex, view, pixel_spacing], axis=1)

    # def to_one_hot(self, x):
    #     label = x.split('|')
    #     label_idx = [self.CLASSES.index(l) for l in label]

    #     out = np.zeros(15)
    #     out[label_idx] = 1

    #     return out