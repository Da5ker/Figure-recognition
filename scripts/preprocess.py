import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset

class DataPreprocessing():
    def __init__(self, images, labels, size=32, color='RGB', test_size=0.2, random_seed=1):
        self.images = images
        self.labels = labels
        self.size = size
        self.color = color
        self.test_size = test_size
        self.random_seed = random_seed
        if color == 'RGB':
            self.pre_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(size, size)),
                transforms.ToTensor()])
        if color == 'BW':
            self.pre_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(size, size)),
                transforms.Grayscale(),
                transforms.ToTensor()])
            
    def scale(self):
        labels = self.labels

        scaler_shape = preprocessing.LabelEncoder().fit(labels['figure'])
        labels['figure'] = scaler_shape.transform(labels['figure'])

        scaler_color = preprocessing.LabelEncoder().fit(labels['color'])
        labels['color'] = scaler_color.transform(labels['color'])

        scaler_size = preprocessing.MinMaxScaler(feature_range=(0, 2)).fit(labels[['size']])
        labels['size'] = scaler_size.transform(labels[['size']])

        scaler_angle = preprocessing.MinMaxScaler(feature_range=(0, 5)).fit(labels[['rotation_angle']])
        labels['rotation_angle'] = scaler_angle.transform(labels[['rotation_angle']])

        scaler_xcoord = preprocessing.MinMaxScaler(feature_range=(0, 2)).fit(labels[['center_x']])
        labels['center_x'] = scaler_xcoord.transform(labels[['center_x']])

        scaler_ycoord = preprocessing.MinMaxScaler(feature_range=(0, 2)).fit(labels[['center_y']])
        labels['center_y'] = scaler_ycoord.transform(labels[['center_y']])
        
        return labels, scaler_shape, scaler_color, scaler_size, scaler_angle, scaler_xcoord, scaler_ycoord

    def split(self, scaler=scale):
        labels = scaler(self)[0]
        X_train, X_test, y_train, y_test = train_test_split(self.images, labels, test_size=self.test_size, random_state=self.random_seed)
        return X_train, X_test, y_train, y_test

    def batch_mean_and_sd(self, scaler=scale):
        labels = scaler(self)[0]
        data = DataLoader(CustomDataset(self.images, labels, transformation=self.pre_transform), batch_size=128, shuffle=False)
        cnt = 0
        fst_moment = torch.empty(1)
        snd_moment = torch.empty(1)

        for images, y1, y2, y3, y4, y5, y6 in data:
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                    dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (
                        cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (
                                cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)        
        return mean, std
    
    def get_transform(self, normalize=batch_mean_and_sd):
        mean, std = normalize(self)
        mean = tuple(mean.tolist())
        std = tuple(std.tolist())
        color = self.color
        size = self.size
        if color == 'RGB':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize((mean), (std))])
        if color == 'BW':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(size, size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((mean), (std))])
        return transform