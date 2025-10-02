import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from PIL import Image

import torch
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from data.cityscapes_labels import CsLabels
from torch.utils.data import Dataset


class CacheCityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = CsLabels.Label
    classes = CsLabels.labels

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        cache_transform: Optional[Callable] = None,
        cache_target_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        sort: bool = False,
        image_paths = None,
        target_paths = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.cache_transform = cache_transform
        self.cache_target_transform = cache_target_transform
        self.images = []
        self.targets = []
        self.cached = False
        self.sort = sort

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = "Unknown value '{}' for argument split if mode is '{}'. Valid values are {{{}}}."
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [
            verify_str_arg(value, "target_type", ("instance", "semantic", "polygon", "color"))
            for value in self.target_type
        ]

        if image_paths is None:
            if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
    
                if split == "train_extra":
                    image_dir_zip = os.path.join(self.root, "leftImg8bit_trainextra.zip")
                else:
                    image_dir_zip = os.path.join(self.root, "leftImg8bit_trainvaltest.zip")
    
                if self.mode == "gtFine":
                    target_dir_zip = os.path.join(self.root, f"{self.mode}_trainvaltest.zip")
                elif self.mode == "gtCoarse":
                    target_dir_zip = os.path.join(self.root, f"{self.mode}.zip")
    
                if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                    extract_archive(from_path=image_dir_zip, to_path=self.root)
                    extract_archive(from_path=target_dir_zip, to_path=self.root)
                else:
                    raise RuntimeError(
                        "Dataset not found or incomplete. Please make sure all required folders for the"
                        ' specified "split" and "mode" are inside the "root" directory'
                    )
    
            for city in tqdm(os.listdir(self.images_dir)):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in tqdm(os.listdir(img_dir)):
                    target_types = []
                    for t in self.target_type:
                        target_name = "{}_{}".format(
                            file_name.split("_leftImg8bit")[0], self._get_target_suffix(self.mode, t)
                        )
                        target_types.append(os.path.join(target_dir, target_name))
    
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(target_types)
                    
        # Build dataset from a precomputed list of images and targets            
        else:
            self.images = image_paths
            self.targets = target_paths
            
        if self.sort:
            self.images, self.targets = (list(t) for t in zip(*sorted(zip(self.images, self.targets))))
            
        self.image_paths = self.images.copy()
        self.target_paths = self.targets.copy()
           
    # Subset the dataset to the given indices        
    def subset(self, indices):
        self.images = [self.images[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        self.image_paths = [self.image_paths[i] for i in indices]
        self.target_paths = [self.target_paths[i] for i in indices]
        
    # Preload images and labels in cache for faster training
    def cache(self):
        
        for ind, (image_path, target_type) in tqdm(enumerate(zip(self.images, self.targets))):
        
            image = Image.open(image_path).convert("RGB")
            
            targets: Any = []
            for i, t in enumerate(self.target_type):
                if t == "polygon":
                    target = self._load_json(target_type[i])
                else:
                    target = Image.open(target_type[i])
    
                targets.append(target)
    
            target = tuple(targets) if len(targets) > 1 else targets[0]
    
            if self.cache_transform is not None:
                image = self.cache_transform(image)
            if self.cache_target_transform is not None:
                target = self.cache_target_transform(target)
                
            self.images[ind] = image
            self.targets[ind] = target
            
        self.cached = True
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """
        if not self.cached:
            image = Image.open(self.images[index]).convert("RGB")
            
            targets: Any = []
            for i, t in enumerate(self.target_type):
                if t == "polygon":
                    target = self._load_json(self.targets[index][i])
                else:
                    target = Image.open(self.targets[index][i])
    
                targets.append(target)
    
            target = tuple(targets) if len(targets) > 1 else targets[0]
        else:  
            image = self.images[index]
            target = self.targets[index]
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"
        
        
        
        
        
class CacheGTA(VisionDataset):

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = CsLabels.Label
    classes = CsLabels.labels

    def __init__(
        self,
        root: str,
        split: str = "train",
        cache_transform: Optional[Callable] = None,
        cache_target_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        sort: bool = False,
        image_paths = None,
        target_paths = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        if split not in ["train", "val", "test"]:
            raise RuntimeError(
                "Not a valid split"
            )
            
        self.images_dir = os.path.join(self.root, "images", split)
        self.targets_dir = os.path.join(self.root, "labels", split)
        self.cache_transform = cache_transform
        self.cache_target_transform = cache_target_transform
        self.images = []
        self.targets = []
        self.cached = False
        self.sort = sort

        if image_paths is None:
            
            if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
                raise RuntimeError(
                    "Dataset not found or incomplete. Please make sure all required folders for the"
                    ' specified "split" and "mode" are inside the "root" directory'
                )

            for file_name in tqdm(os.listdir(self.images_dir)):
                
                # Preload images and labels in cache for faster training
                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(os.path.join(self.targets_dir, file_name))
        else:
            self.images = image_paths
            self.targets = target_paths

        if self.sort:
            self.images, self.targets = (list(t) for t in zip(*sorted(zip(self.images, self.targets))))
            
        print(self.images[:10], self.targets[:10])

        self.image_paths = self.images.copy()
        self.target_paths = self.targets.copy()
        
    # Subset the dataset to the given indices 
    def subset(self, indices):
        self.images = [self.images[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        self.image_paths = [self.image_paths[i] for i in indices]
        self.target_paths = [self.target_paths[i] for i in indices]
        
    # Preload images and labels in cache for faster training
    def cache(self):
        
        for ind, (image_path, target_path) in tqdm(enumerate(zip(self.images, self.targets))):
        
            image = Image.open(image_path).convert("RGB")
            target = Image.open(target_path).convert("RGB")

            if self.cache_transform is not None:
                image = self.cache_transform(image)
            if self.cache_target_transform is not None:
                target = self.cache_target_transform(target)
                
            self.images[ind] = image
            self.targets[ind] = target  
            
        self.cached = True
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = self.images[index]
        target = self.targets[index]
        
        if not self.cached:
            image = Image.open(image).convert("RGB")
            target = Image.open(target).convert("RGB")
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
    
    
    
# ------------------ Cityscapes + GTA source classification -----------------

class CacheCityscapesClassification(VisionDataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        label: int = 0,
        cache_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        sort: bool = False,
        image_paths = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.split = split
        self.label = label
        self.cache_transform = cache_transform
        self.images = []
        self.cached = False
        self.sort = sort

        valid_modes = ("train", "test", "val")
        msg = "Unknown value '{}' for argument split if mode is '{}'. Valid values are {{{}}}."
        verify_str_arg(split, "split", valid_modes, msg)

        if image_paths is None:
            if not os.path.isdir(self.images_dir):
    
                if split == "train_extra":
                    image_dir_zip = os.path.join(self.root, "leftImg8bit_trainextra.zip")
                else:
                    image_dir_zip = os.path.join(self.root, "leftImg8bit_trainvaltest.zip")
    
                if os.path.isfile(image_dir_zip):
                    extract_archive(from_path=image_dir_zip, to_path=self.root)
                else:
                    raise RuntimeError(
                        "Dataset not found or incomplete. Please make sure all required folders for the"
                        ' specified "split" and "mode" are inside the "root" directory'
                    )
    
            for city in tqdm(os.listdir(self.images_dir)):
                img_dir = os.path.join(self.images_dir, city)
                for file_name in tqdm(os.listdir(img_dir)):
                    self.images.append(os.path.join(img_dir, file_name))
                    
        # Build dataset from a precomputed list of images and targets            
        else:
            self.images = image_paths
            
        if self.sort:
            self.images = sorted(self.images)
            
        self.labels = [self.label]*len(self.images)
        self.image_paths = self.images.copy()
           
    # Subset the dataset to the given indices        
    def subset(self, indices):
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.image_paths = [self.image_paths[i] for i in indices]
        
    # Preload images and labels in cache for faster training
    def cache(self):
        
        for ind, image_path in tqdm(enumerate(self.images)):
        
            image = Image.open(image_path).convert("RGB")
            
            if self.cache_transform is not None:
                image = self.cache_transform(image)
                
            self.images[ind] = image
            
        self.cached = True
       
    # Enable to change the label for certain images
    def change_labels(self, indices, new_label):
        
        for i in indices:
            self.labels[i] = new_label
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """
        if not self.cached:
            image = Image.open(self.images[index]).convert("RGB")
        else:  
            image = self.images[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        target = self.labels[index]

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data
        
        
        
class CacheGTAClassification(VisionDataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        label: int = 1,
        cache_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        sort: bool = False,
        image_paths = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        if split not in ["train", "val", "test"]:
            raise RuntimeError(
                "Not a valid split"
            )
            
        self.images_dir = os.path.join(self.root, "images", split)
        self.label = label
        self.cache_transform = cache_transform
        self.images = []
        self.cached = False
        self.sort = sort

        if image_paths is None:
            
            if not os.path.isdir(self.images_dir):
                raise RuntimeError(
                    "Dataset not found or incomplete. Please make sure all required folders for the"
                    ' specified "split" and "mode" are inside the "root" directory'
                )

            for file_name in tqdm(os.listdir(self.images_dir)):
                
                # Preload images and labels in cache for faster training
                self.images.append(os.path.join(self.images_dir, file_name))
        else:
            self.images = image_paths

        if self.sort:
            self.images = sorted(self.images)
            
        print(self.images[:10])
        
        self.labels = [self.label]*len(self.images)
        self.image_paths = self.images.copy()
        
    # Subset the dataset to the given indices 
    def subset(self, indices):
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.image_paths = [self.image_paths[i] for i in indices]
        
    # Preload images and labels in cache for faster training
    def cache(self):
        
        for ind, image_path in tqdm(enumerate(self.images)):
        
            image = Image.open(image_path).convert("RGB")

            if self.cache_transform is not None:
                image = self.cache_transform(image)
                
            self.images[ind] = image
            
        self.cached = True
       
    # Enable to change the label for certain images
    def change_labels(self, indices, new_label):
        
        for i in indices:
            self.labels[i] = new_label
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = self.images[index]
        
        if not self.cached:
            image = Image.open(image).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)

        target = self.labels[index]
        
        return image, target

    def __len__(self) -> int:
        return len(self.images)
    
    
# ----------------------- Triple MNIST datasets -------------------------    
class TripleMNISTClassificationDataset(Dataset):
    
    # Classes 0-9 digits
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)/255.
        self.labels = torch.tensor([[int(c in l) for c in "0123456789"] for l in labels], dtype=torch.uint8)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    
class TripleMNISTSegmentationDataset(Dataset):
    
    def __init__(self, images, masks, transform=None, target_transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.images = (self.images - torch.min(self.images)) / (torch.max(self.images) - torch.min(self.images))
        self.masks = torch.tensor(masks, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return image, mask
    
    
class TripleMNIST_Source_Classifier_Dataset(Dataset):
    
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.images = (self.images - torch.min(self.images)) / (torch.max(self.images) - torch.min(self.images))
        
        self.labels = torch.tensor(labels, dtype=torch.uint8)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label