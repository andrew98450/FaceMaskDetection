import torch
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.ImageFolder(
    root="Dataset/Train/", transform=transform)

class FaceMaskNet(torch.nn.Module):
    
    def __init__(self):
        
        super(FaceMaskNet, self).__init__()
        
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2))
        
        self.conv2_layer = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2))
        
        self.conv3_layer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),                
            torch.nn.MaxPool2d(2, stride=2))
        
        self.conv4_layer = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),     
            torch.nn.MaxPool2d(2, stride=2))      
        
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 2 * 2, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 2),                 
            torch.nn.Softmax(1))
        
    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.conv2_layer(output)
        output = self.conv3_layer(output)
        output = self.conv4_layer(output)
        
        output = self.fc_layer(output)
        return output