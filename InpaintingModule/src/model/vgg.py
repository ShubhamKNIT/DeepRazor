import torch
import torch.nn as nn
from torchvision.models import vgg16

class VGG16PerceptualFeatures(nn.Module):
    def __init__(self):
        super(VGG16PerceptualFeatures, self).__init__()
        # Load the pretrained VGG16 model
        vgg = vgg16(weights='IMAGENET1K_V1')
        # Extract features up to relu4_3
        self.features = nn.Sequential(*list(vgg.__getattr__('features')[:23]))  # relu4_3 is at index 22
        # Freeze the weights of the model
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

