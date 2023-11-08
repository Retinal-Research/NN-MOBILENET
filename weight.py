import os
import cv2
import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from mobileNet import ReXNetV1


model = ReXNetV1(width_mult=3.0, classes=5)
checkpoint_path = 'fold_4_b.pth'
        
wegiht = torch.load(checkpoint_path, map_location='cpu')['model']
#wegiht = torch.load(checkpoint_path, map_location='cpu')

model.load_state_dict(wegiht,strict=True)

torch.save(model.state_dict(),'fold_4_b_new.pth')