import os
import sys

from pprintpp import pprint

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset
import argparse
import json
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from torchsummary import summary
import monai

from efficientvit import seg_model_zoo
from efficientvit.models.utils import resize

#from ..utils.selection_utils import get_model, get_transform, get_loss_metric
from plot_utils import imshow
from sklearn.metrics import classification_report
from torchvision.utils import draw_segmentation_masks
from data.datasets import CacheCityscapesClassification, CacheGTAClassification
from data.transforms import prepare_plot_im, prepare_plot_label, \
    generate_transform_cityscapes_im, generate_transform_cityscapes_label, generate_transform_GTA5_label
from models.models import modified_get_conv_layer
from metrics.metrics import CumulativeClassificationMetrics, CumulativeIoUCityscapes
from models.models import CNN_GTA_Cityscapes_source_Classification

import psutil


def main(args, config):

    # Initializing tensorboard summary
    log_dir = config["log_dir"] + "/" + os.path.basename(__file__)
    if config["log_filename"] is not None:
        log_dir += "/"+config["log_filename"]
    else:
        log_dir += "/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(log_dir=log_dir)
    
    # Copy config file in the experiment folder
    shutil.copy(args.config_path, log_dir)
    
    # Get transforms
    transform = generate_transform_cityscapes_im(size=config["size"])
    
    # Get train and validation sets, datasets and loaders
    ds_city = CacheCityscapesClassification(config["data_dir_cityscapes"], split='test', transform=transform, sort=config["sort"])   
    ds_gta = CacheGTAClassification(config["data_dir_gta"], split='test', transform=transform, sort=config["sort"])
    print(len(ds_city), len(ds_city[0]), ds_city.__getitem__(0))
    
    # Equilibrate the number of samples from cityscapes and GTA5 in the final training set.
    dataset_size = len(ds_city)
    
    if config["cache"]:
        # Preload images and labels in cache to accelerate training
        ds_city.cache()
        ds_gta.cache()

    for i in range(len(ds_city)):
        fig = plt.figure()
        plt.imshow(prepare_plot_im(ds_city[i][0]))
        plt.title(f"Cityscapes {ds_city[i][1]}")
        plt.tight_layout()
        writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
        if i > 10:
            break
    for i in range(len(ds_gta)):
        fig = plt.figure()
        plt.imshow(prepare_plot_im(ds_gta[i][0]))
        plt.title(f"GTA5 {ds_gta[i][1]}")
        plt.tight_layout()
        writer.add_figure(f"Verification_GTA", fig, global_step=i)
        if i > 10:
            break
        
    # Build the final training and validation sets with loaders
    data = ConcatDataset([ds_city, ds_gta])
    loader = torch.utils.data.DataLoader(data, batch_size=1, 
                                         shuffle=False, num_workers=4)
    
    im, labels = ds_city[0]
    print(im.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(ds_city))
    
    im, labels = ds_gta[0]
    print(im.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(ds_gta))
    
    im, labels = next(iter(loader))
    print(im.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(data))

    device = torch.device("cuda:0")
    
    source_classifier = CNN_GTA_Cityscapes_source_Classification(sources=2).to(device)
    source_classifier.load_state_dict(torch.load(config["source_classifier_path"]))

    # Get losses, metrics and optimizers
    source_classifier.eval()
        
    with torch.no_grad():
        
        # Main loop on validation set
        for (idx, batch_data) in enumerate(tqdm(loader)):
            
            inp, label = batch_data[0].to(device), batch_data[1].to(device)
            
            source_classif_output = source_classifier(inp)
            _, source_prediction = torch.max(source_classif_output, 1)
            
            if config["invert_classes"]:
                source_prediction = 1 - source_prediction
                
            if idx == 0:
                total_truth = label.detach().cpu().numpy()
                total_pred = source_prediction.detach().cpu().numpy()
            else:
                total_truth = np.concatenate([total_truth, label.detach().cpu().numpy()], axis=0)
                total_pred = np.concatenate([total_pred, source_prediction.detach().cpu().numpy()], axis=0)

        report_dict = classification_report(total_truth, total_pred, target_names=["Cityscapes", "GTA"], output_dict=True)

        with open(os.path.join(log_dir, 'val_perf.json'), 'w') as f:
            json.dump(report_dict, f, indent=4)  
            
        print(report_dict)
    
if __name__ == "__main__":
        
    print("Curent working directory: ", os.getcwd())
    
    print("Is cuda avaiable? ", torch.cuda.is_available())
    print("Number of cuda devices available: ", torch.cuda.device_count())
    
    # Define argument parser and its attributes
    parser = argparse.ArgumentParser(description='CIFAR 10 classif')
    
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='path to json config file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    main(args, config)
