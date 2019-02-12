import torch
from torch import nn
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from skimage.transform import resize

class Encoder(nn.Module):
    """
    Encoder class to transform images into features. We're using the features from the last conv layer.
    So the output will be 14x14x512
    """

    def __init__(self):

        super(Encoder, self).__init__()

        # Imgs must be normalised before being fed to the CNN as per de docs
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform=transforms.Compose([normalize])
        
        # Using the official VGG16 model pretrained
        vgg = torchvision.models.vgg16(pretrained=True)  
        
        # Remove the classification bloc
        modules = list(vgg.children())[:-1]

        # Remove the last maxpool layer
        modules[0]=nn.Sequential(*list(modules[0].children())[:-1])

        # Save the new model 
        self.vgg = nn.Sequential(*modules)


    def forward(self, images):
        """
        Forward propagation.

        Params:
            - images : tensor of input images (batch_size, 3, 224, 224)
        Return:
            - output : Tensor of features (14,14,512)
        """
        output = self.transform(images.squeeze())
        output = self.vgg(images)  
        output = output.permute(0, 2, 3, 1)  
        return output


if __name__ == "__main__":

    # Small test to check if vgg is working correctly
    encoder=Encoder()
    print(encoder)
    image=io.imread('./raw_data/Flickr8k_data/667626_18933d713e.jpg')
    image = resize(image, (224, 224))
    image=torch.Tensor(image)
    image=image.permute(2,0,1).reshape(1,3,image.shape[0],image.shape[1])
    print(encoder.forward(image).shape)


