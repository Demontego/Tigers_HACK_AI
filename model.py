import torch
import torch.nn as nn
from torchvision import models


class img_CNN(nn.Module):
    def __init__(self, model_type, num_classes):
        super(img_CNN, self).__init__()
        self.model_type = model_type
        if model_type == 'ResNext':
            self.model = models.resnext50_32x4d(pretrained=True)
        elif model_type == 'ResNet':
            self.model = models.resnet18(pretrained=True)
        elif model_type == 'DenseNet':
            self.model = models.densenet161(pretrained=True)
        elif model_type == 'GoogleNet':
            self.model = models.googlenet(pretrained=True)
        elif model_type== 'MobileNet':
            self.model = models.mobilenet_v3_small(pretrained=True)
        elif model_type == 'Inception':
            self.model = models.inception_v3(pretrained=True)
        elif model_type == 'Wide ResNet':
            self.model = models.wide_resnet50_2(pretrained=True)
        else:
            raise ValueError('Wrong model type!')
        self.conc_models = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(1000, 128), 
                                             nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, image):
        img_feature = self.model(image)
        img_feature = self.conc_models(img_feature)
        return img_feature
    
class img_siamese(nn.Module):
    def __init__(self, model_type):
        super(img_siamese, self).__init__()
        self.model_type = model_type
        if model_type == 'ResNext':
            self.model = models.resnext50_32x4d(pretrained=True)
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 512))
        elif model_type == 'ResNet':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256))
        elif model_type == 'DenseNet':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Sequential(nn.Linear(self.model.classifier.in_features, 256))
        elif model_type == 'GoogleNet':
            self.model = models.googlenet(pretrained=True)
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256))
        else:
            raise ValueError('Wrong model type!')
        self.linear = nn.Linear(512,2)
        self.sigmoid = nn.Sigmoid()
        
    def convs(self, x):
        img_feature = self.model(x)
        return img_feature
        
    def forward(self, image1, image2):
        img_feature1 = self.convs(image1)
        img_feature2 = self.convs(image2)
        img_feature1 = self.sigmoid(img_feature1)
        img_feature2 = self.sigmoid(img_feature2)
        feature = torch.abs(img_feature1-img_feature2)
        feature = self.linear(feature)
        return feature