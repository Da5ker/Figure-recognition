import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        self.convolutional_layer_1 = nn.Sequential(            
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.2))
        
        self.convolutional_layer_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.2))
        
        self.convolutional_layer_3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.2))

        self.linear_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=128*4*4, out_features=100),
                nn.ReLU())
        
        self.output_shape = nn.Linear(in_features=100, out_features=5)

        self.output_color = nn.Linear(in_features=100, out_features=10)
        
        self.output_size = nn.Sequential(
                nn.Linear(in_features=100, out_features=50),
                nn.ReLU(),
                nn.Linear(in_features=50, out_features=1))
        
        self.output_angle = nn.Sequential(
                nn.Linear(in_features=100, out_features=50),
                nn.ReLU(),
                nn.Linear(in_features=50, out_features=1))

        self.output_xcoord = nn.Sequential(
                nn.Linear(in_features=100, out_features=50),
                nn.ReLU(),
                nn.Linear(in_features=50, out_features=1))
        
        self.output_ycoord = nn.Sequential(
                nn.Linear(in_features=100, out_features=50),
                nn.ReLU(),
                nn.Linear(in_features=50, out_features=1))

    def forward(self, x):
        # # print(x.shape)
        x = self.convolutional_layer_1(x)
        # print(x.shape)
        x = self.convolutional_layer_2(x)
        # print(x.shape)
        x = self.convolutional_layer_3(x)
        # print(x.shape)
        x = self.linear_layer(x)
        y_shape = self.output_shape(x)
        y_color = self.output_color(x)
        y_size = self.output_size(x)
        y_angle = self.output_angle(x)
        y_xcoord = self.output_xcoord(x)
        y_ycoord = self.output_ycoord(x)
        return y_shape, y_color, y_size, y_angle, y_xcoord, y_ycoord