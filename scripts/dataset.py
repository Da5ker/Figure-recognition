import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels, transformation=None):
        self.images = images
        self.shape = labels['figure'].values
        self.color = labels['color'].values
        self.size = labels['size'].values
        self.angle = labels['rotation_angle'].values
        self.xcoord = labels['center_x'].values
        self.ycoord = labels['center_y'].values
        self._transform = transformation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        shape = self.shape[index]
        color = self.color[index]
        size = self.size[index]
        angle = self.angle[index]
        xcoord = self.xcoord[index]
        ycoord = self.ycoord[index]
        
        if self._transform:
            image = self._transform(image)

        return image, shape.astype('int64'), color.astype('int64'), size.astype('float32'), angle.astype('float32'), xcoord.astype('float32'), ycoord.astype('float32')