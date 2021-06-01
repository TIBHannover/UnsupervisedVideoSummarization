import torchvision.models as models
import torchvision as tv
import torch
import torch.nn as nn



class FeatureExtractor(nn.Module):
    """Class is responsible for extracting deep features of a video frame (image)
    """    
    def __init__(self, arch):
        super(FeatureExtractor, self).__init__()
        # set model architecture according to architecture input name
        self.set_model_arch(arch)
        # resize frame and normalize
        self.tranform = tv.transforms.Compose([
            tv.transforms.Resize([224, 224]), tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        # get the pool layer of the model
        self.model = nn.Sequential(*(list(self.arch.children())[:-2] + [nn.MaxPool2d(1, 1)]))

    def forward(self, frame):
        features = self.model(frame)
        features = features.reshape((features.shape[0], -1))
        return features

    def set_model_arch(self, arch):
        # model architecture
        if arch == 'alexnet':
            self.arch = models.alexnet(pretrained=True)
        elif arch == 'resnet50':
            self.arch = models.resnet50(pretrained=True)
        elif arch == 'resnet152':
            self.arch = models.resnet152(pretrained=True)
        else:
            self.arch = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
