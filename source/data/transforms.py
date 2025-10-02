import numpy as np
from monai.transforms import (
    Transform,
    MapTransform,
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    LoadImage,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    Spacingd,
    SpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClassesd,
    DataStatsd,
    RandGaussianNoised,
    SqueezeDimd,
    CropForegroundd,
    SpatialPadd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    FromMetaTensord,
    ToTensord,
    CenterSpatialCropd,
    SqueezeDimd,
    MaskIntensityd,
    DataStatsd,
    NormalizeIntensity,
    ToTensor
)
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from data.cityscapes_labels import CsLabels
import torch.nn.functional as F


# -------------------------- Cityscapes / GTA5 ------------------------

def generate_transform_cityscapes_im(size=[256, 512], imagewise_standard=False, normalization="imagenet"):
        
    trans = [
        transforms.ToTensor(),
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR)
    ]
    
    if imagewise_standard:
        trans += [NormalizeIntensity(), ToTensor()]
        
    elif normalization == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        trans += [transforms.Normalize(mean, std)]
        
    elif normalization == "cityscapes":
        mean = (0.29866842, 0.30135223, 0.30561872)
        std = (0.23925215, 0.23859318, 0.2385942)
        trans += [transforms.Normalize(mean, std)]
        
    return transforms.Compose(trans)


def generate_transform_cityscapes_label(size=[256, 512], labels="trainId"):
    
    composed_transforms = transforms.Compose([
        transforms.PILToTensor(),
        convert_labels_cityscapes(labels=labels),
        transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    ])

    return composed_transforms


def generate_transform_GTA5_label(size=[256, 512], labels="trainId"):
    
    composed_transforms = transforms.Compose([
        transforms.PILToTensor(),
        convert_labels_GTA5(labels=labels),
        transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    ])

    return composed_transforms


def prepare_plot_im(im, normalization="imagenet"):
    
    if normalization == "imagenet":
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=im.dtype, device=im.device).view(-1, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=im.dtype, device=im.device).view(-1, 1, 1)
        
    elif normalization == "cityscapes":
        mean = torch.as_tensor([0.29866842, 0.30135223, 0.30561872], dtype=im.dtype, device=im.device).view(-1, 1, 1)
        std = torch.as_tensor([0.23925215, 0.23859318, 0.2385942], dtype=im.dtype, device=im.device).view(-1, 1, 1)
        
    return im.mul(std).add(mean).permute((1,2,0)).detach().cpu().numpy()
    

def prepare_plot_label(seg_map, labels="trainId"):
    
    color_label = np.zeros((seg_map.shape[0], seg_map.shape[1], 3))
    seg = seg_map.detach().cpu().numpy()
    
    if labels == "trainId":
        for label in reversed(CsLabels.labels):
            color_label[seg == label.trainId] = label.color
    elif labels == "categories":
        for label in reversed(CsLabels.labels):
            color_label[seg == label.categoryId] = label.color
        
    color_label /= 255.0
    return color_label
    
    
class convert_labels_cityscapes():
    
    def __init__(self, labels="trainId"):
        self.labels = labels
        
    def __call__(self, seg_map):
        
        if self.labels == "trainId":
            trainId_map = torch.ones(seg_map.shape, dtype=torch.uint8)*255
            for label in CsLabels.labels:
                trainId_map[seg_map == label.id] = label.trainId
                
        elif self.labels == "categories":
            trainId_map = torch.ones(seg_map.shape, dtype=torch.uint8)*255
            for label in CsLabels.labels:
                trainId_map[seg_map == label.id] = label.categoryId

        return trainId_map


class convert_labels_GTA5():
    
    def __init__(self, labels="trainId"):
        self.labels = labels
        
    def __call__(self, seg_map):
        seg_map = seg_map.permute((1, 2, 0))

        if self.labels == "trainId":
            trainId_map = torch.ones(seg_map.shape[:2], dtype=torch.uint8)*255
            for label in CsLabels.labels:
                if label.trainId not in [-1, 255]:
                    trainId_map[(seg_map == torch.tensor(label.color)).all(dim=-1)] = label.trainId
                    
        elif self.labels == "categories":
            trainId_map = torch.ones(seg_map.shape[:2], dtype=torch.uint8)*255
            for label in CsLabels.labels:
                trainId_map[(seg_map == torch.tensor(label.color)).all(dim=-1)] = label.categoryId
                    
        return trainId_map.unsqueeze(0)
    

