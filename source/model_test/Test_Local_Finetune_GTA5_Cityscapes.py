import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd
import matplotlib
matplotlib.use('Agg')

import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset
import os
import argparse
import json
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from torchsummary import summary
import monai
from monai.networks.nets import DynUNet
from efficientvit import seg_model_zoo
from efficientvit.models.utils import resize
from sklearn.metrics.cluster import rand_score
import gc

from pprintpp import pprint

#from ..utils.selection_utils import get_model, get_transform, get_loss_metric

from plot_utils import imshow
from torchvision.utils import draw_segmentation_masks
from data.cityscapes_labels import CsLabels
from data.datasets import CacheCityscapes, CacheGTA
from data.transforms import prepare_plot_im, prepare_plot_label, \
    generate_transform_cityscapes_im, generate_transform_cityscapes_label, generate_transform_GTA5_label
from models.models import modified_get_conv_layer
from metrics.metrics import CumulativeClassificationMetrics, CumulativeIoUCityscapes, sample_mIoU_Cityscapes
from data.data_utils import recursive_add_parent_in_path, merge_inst_cluster_part
from data.partitioning_utils import GTA_Cityscapes_iid, GTA_Cityscapes_Full_non_iid, GTA_Cityscapes_Dirichlet_non_iid, assert_GTA_Cityscapes_partitioning
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
    val_transform_city = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_cityscapes_label(size=config["size"], labels=config["labels"]))
    val_transform_gta = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_GTA5_label(size=config["size"], labels=config["labels"]))
    
    # Get train and validation sets, datasets and loaders
    batch_size = config["batch_size"]

    if config["cache"]:
        val_ds_city = CacheCityscapes(config["data_dir_cityscapes"], split=config["split"], mode='fine', target_type='semantic', 
                                 cache_transform=val_transform_city[0], cache_target_transform=val_transform_city[1], sort=config["sort"])   
        val_ds_gta = CacheGTA(config["data_dir_gta"], split=config["split"],
                                 cache_transform=val_transform_gta[0], cache_target_transform=val_transform_gta[1], sort=config["sort"])
    else:
        val_ds_city = CacheCityscapes(config["data_dir_cityscapes"], split=config["split"], mode='fine', target_type='semantic', 
                                 transform=val_transform_city[0], target_transform=val_transform_city[1], sort=config["sort"])   
        val_ds_gta = CacheGTA(config["data_dir_gta"], split=config["split"],
                                 transform=val_transform_gta[0], target_transform=val_transform_gta[1], sort=config["sort"])
    print(len(val_ds_city), len(val_ds_city[0]), val_ds_city.__getitem__(0))
    
    # Equilibrate the number of samples from cityscapes and GTA5 in the final training set.
    min_dataset_size_val = min(len(val_ds_city), len(val_ds_gta))
    
    print(min_dataset_size_val)
    
    if config["cache"]:
        # Preload images and labels in cache to accelerate training
        val_ds_city.cache()
        val_ds_gta.cache()
        
    for i in range(len(val_ds_city)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(val_ds_city[i][0]))
        axs[1].imshow(prepare_plot_label(val_ds_city[i][1][0], labels=config["labels"]), alpha=1.)
        plt.title("Cityscapes")
        plt.tight_layout()
        writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
        if i > 10:
            break
    for i in range(len(val_ds_gta)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(val_ds_gta[i][0]))
        axs[1].imshow(prepare_plot_label(val_ds_gta[i][1][0], labels=config["labels"]), alpha=1.)
        plt.title("GTA5")
        plt.tight_layout()
        writer.add_figure(f"Verification_GTA", fig, global_step=i)
        if i > 10:
            break
      
    # Build the final training and validation sets with loaders
    val_data = ConcatDataset([val_ds_city, val_ds_gta])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, 
                                              shuffle=False, num_workers=4)
    
    val_loader_city = torch.utils.data.DataLoader(val_ds_city, batch_size=batch_size, 
                                              shuffle=False, num_workers=0)
    val_loader_gta = torch.utils.data.DataLoader(val_ds_gta, batch_size=batch_size, 
                                              shuffle=False, num_workers=0)

    """
    for im, labels in train_ds_dict["0"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["0"]["0"]))
    
    for im, labels in train_ds_dict["1"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["1"]["0"]))
    """

    im, labels = next(iter(val_loader))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    
    if config["labels"] == "trainId":
        ignore_index = 255
        nb_classes = 20
        ignore_background = True
        
    elif config["labels"] == "categories":
        ignore_index = 255
        nb_classes = 8
        ignore_background = True
        
    num_classes = nb_classes - ignore_background
    label_values = list(range(num_classes))
    
    device = torch.device("cuda:0")
    
    clients = [str(c) for c in list(range(config["nb_clients"]))]
    
    if config["generate_split"]:
        
        size = min_dataset_size_val
            
        if config["iid"]:
            if not os.path.exists("../../datasets/Cityscapes_test/IID_partitioning"):
                os.mkdir("../../datasets/Cityscapes_test/IID_partitioning")
            val_local_samples = GTA_Cityscapes_iid(size, clients, save_folder="../../datasets/Cityscapes_test/IID_partitioning", seed=0, split=config["split"])
            
        else:
            if config["dirichlet"] == 0:
                if not os.path.exists("../../datasets/Cityscapes_test/Full_non_IID_partitioning"):
                    os.mkdir("../../datasets/Cityscapes_test/Full_non_IID_partitioning")
                val_local_samples = GTA_Cityscapes_Full_non_iid(size, clients, save_folder="../../datasets/Cityscapes_test/Full_non_IID_partitioning", seed=1, split=config["split"])
                
            else:
                # dir 0.25: seed 6 ok, seed 5 less homogeneous but more clustered, seed 20 best so far
                # dir 0.4: seed 20 ok but ùaybe to homogeneous, seed 18 ok
                if not os.path.exists("../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning"):
                    os.mkdir("../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning")
                val_local_samples = GTA_Cityscapes_Dirichlet_non_iid(size, clients, config["dirichlet"], save_folder="../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning", seed=20, split=config["split"])
    else:
        with open(os.path.join(config["partitioning_dir"], config["partition_file"]), 'r') as f:
            val_local_samples = json.load(f)

    clients = list(val_local_samples.keys())
        
    # Assure that we use a correct partitioning of the data
    assert_GTA_Cityscapes_partitioning(val_local_samples, clients)
    
    # Save the partitioning
    with open(os.path.join(log_dir, 'part_val_samples.json'), 'w') as f:
        json.dump(val_local_samples, f, indent=4)

    # In a local finetuning setup, needs some local validation sets
    val_local_datasets = {} 
    for c in clients:
        if "Cityscapes" not in val_local_samples[c]:
            val_local_datasets[c] = torch.utils.data.Subset(val_ds_gta, val_local_samples[c]["GTA"])
        elif "GTA" not in val_local_samples[c]:
            val_local_datasets[c] = torch.utils.data.Subset(val_ds_city, val_local_samples[c]["Cityscapes"])
        else:
            val_local_datasets[c] = ConcatDataset([
                torch.utils.data.Subset(val_ds_city, val_local_samples[c]["Cityscapes"]),
                torch.utils.data.Subset(val_ds_gta, val_local_samples[c]["GTA"])
            ])
        
    val_local_loaders = {c:torch.utils.data.DataLoader(val_local_datasets[c], batch_size=batch_size, shuffle=True, num_workers=0) for c in clients}
    
    example_client = clients[0]
        
    if config["model"] == "EfficientVIT":
        # Build the model and its parallel counterpart
        if config["labels"] == "trainId":
            local_models = {client:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for client in clients}
            
        elif config["labels"] == "categories":
            local_models = {client:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for client in clients}

    for client in clients:
        local_models[client].load_state_dict(torch.load(os.path.join(config['perso_models_dir'], config["model_file"].replace("#", client))))
        
    print(local_models[example_client])
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(local_models[example_client], next(iter(val_loader))[0].to(device))
    print(next(iter(val_loader))[0].size())
    #summary(cluster_models["0"], next(iter(train_loader_dict["0"]["0"]))[0][0].size())

    # Get losses, metrics and optimizers
    metric_function = CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)
    
    for client in clients:
        
        local_models[client].eval()
        
        with torch.no_grad():
            
            for (idx, batch_data) in enumerate(tqdm(val_local_loaders[client])):
                
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                outputs = local_models[client](inputs)
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = resize(outputs, size=labels.shape[-2:])
                _, prediction = torch.max(outputs.data, 1, keepdim=True)
                
                metric_function(prediction, labels)

    report = metric_function.aggregate()
    metric_function.reset()
    
    print(report)

    with open(os.path.join(log_dir, f'final_{config["split"]}_inference_results_on_{os.path.basename(config["partitioning_dir"])}.json'), 'w') as fp:
        json.dump(report, fp, indent=4)
        
    with open(os.path.join(config["perso_models_dir"], f'final_{config["split"]}_inference_results_on_{os.path.basename(config["partitioning_dir"])}.json'), 'w') as fp:
        json.dump(report, fp, indent=4)
    
            
    
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
