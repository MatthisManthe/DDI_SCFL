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
from torchvision.utils import draw_segmentation_masks
from data.datasets import CacheCityscapes, CacheGTA
from data.transforms import prepare_plot_im, prepare_plot_label, \
    generate_transform_cityscapes_im, generate_transform_cityscapes_label, generate_transform_GTA5_label
from models.models import modified_get_conv_layer
from metrics.metrics import CumulativeClassificationMetrics, CumulativeIoUCityscapes

import psutil

    
def apply_inference_model(config, batch_data, model, metric_functions, device, idx, writer):
    
    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    outputs = model(inputs)
    if outputs.shape[-2:] != labels.shape[-2:]:
        outputs = resize(outputs, size=labels.shape[-2:])
    _, prediction = torch.max(outputs.data, 1, keepdim=True)
    
    for metric_function in metric_functions:
        metric_function(prediction, labels)
        
    if idx < 5:
        for i in range(len(inputs)):
            fig, axs = plt.subplots(2, 1)
            print(inputs[i].shape, labels[i].shape, prediction[i].shape)
            axs[0].imshow(prepare_plot_im(inputs[i]))
            axs[0].imshow(prepare_plot_label(labels[i][0], labels=config["labels"]), alpha=0.7)
            axs[1].imshow(prepare_plot_im(inputs[i]))
            axs[1].imshow(prepare_plot_label(prediction[i][0], labels=config["labels"]), alpha=0.7)
            plt.tight_layout()
            writer.add_figure(f"Test_prediction", fig, global_step=idx*len(inputs)+i)
        
    return prediction


def aggregate_metrics_and_save(config, metric_functions, best_current, model, log_dir, writer):
        
    report = metric_functions[0].aggregate()
    metric_functions[0].reset()
    
    pprint(report)
    
    writer.add_scalar("Validation/mIoU", report["mIoU"], 1)
    for c, iou in report["IoUs"].items():
        writer.add_scalar(f"Validation/IoU_{c}", iou, 1)
    
    if best_current["mIoU"] < report["mIoU"]:
        
        best_current.update(report) 
        
        print("Saving new best model")
        
        with open(os.path.join(log_dir, f'final_{config["split"]}_{config["model_file"]}_results.json'), 'w') as fp:
            json.dump(best_current, fp, indent=4)
            
        with open(os.path.join(config["model_path"], f'final_{config["split"]}_{config["model_file"]}_results.json'), 'w') as fp:
            json.dump(best_current, fp, indent=4)
    
    return report
               
    

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
    transform_city = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_cityscapes_label(size=config["size"], labels=config["labels"]))
    transform_gta = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_GTA5_label(size=config["size"], labels=config["labels"]))
    
    # Get train and validation sets, datasets and loaders
    if config["cache"]:
        ds_city = CacheCityscapes(config["data_dir_cityscapes"], split=config["split"], mode='fine', target_type='semantic', 
                                 cache_transform=transform_city[0], cache_target_transform=transform_city[1], sort=config["sort"])   
        ds_gta = CacheGTA(config["data_dir_gta"], split=config["split"],
                                 cache_transform=transform_gta[0], cache_target_transform=transform_gta[1], sort=config["sort"])
    else:
        ds_city = CacheCityscapes(config["data_dir_cityscapes"], split=config["split"], mode='fine', target_type='semantic', 
                                 transform=transform_city[0], target_transform=transform_city[1], sort=config["sort"])   
        ds_gta = CacheGTA(config["data_dir_gta"], split=config["split"],
                                 transform=transform_gta[0], target_transform=transform_gta[1], sort=config["sort"])
    print(len(ds_city), len(ds_city[0]), ds_city.__getitem__(0))
    
    # Equilibrate the number of samples from cityscapes and GTA5 in the final training set.
    dataset_size = len(ds_city)
    
    if config["cache"]:
        # Preload images and labels in cache to accelerate training
        ds_city.cache()
        ds_gta.cache()

    for i in range(len(ds_city)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(ds_city[i][0]))
        axs[1].imshow(prepare_plot_label(ds_city[i][1][0], labels=config["labels"]), alpha=1.)
        plt.title(f"Cityscapes {ds_city.image_paths[i]}")
        plt.tight_layout()
        writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
        if i > 10:
            break
    for i in range(len(ds_gta)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(ds_gta[i][0]))
        axs[1].imshow(prepare_plot_label(ds_gta[i][1][0], labels=config["labels"]), alpha=1.)
        plt.title(f"GTA5 {ds_gta.image_paths[i]}")
        plt.tight_layout()
        writer.add_figure(f"Verification_GTA", fig, global_step=i)
        if i > 10:
            break
        
    # Build the final training and validation sets with loaders
    data = ConcatDataset([ds_city, ds_gta])
    loader = torch.utils.data.DataLoader(data, batch_size=1, 
                                         shuffle=False, num_workers=4)
    
    im, labels = ds_city[0]
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(ds_city))
    
    im, labels = ds_gta[0]
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(ds_gta))
    
    im, labels = next(iter(loader))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(data))
    
    if config["labels"] == "trainId":
        ignore_index = 255
        nb_classes = 20
        ignore_background = True
        
    elif config["labels"] == "categories":
        ignore_index = 255
        nb_classes = 8
        ignore_background = True
        
    device = torch.device("cuda:0")

    # Build the model and its parallel counterpart
    if config["labels"] == "trainId":
        model = seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"]).to(device)
    elif config["labels"] == "categories":
        model = seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device)
    
    model.load_state_dict(torch.load(os.path.join(config["model_path"], config["model_file"])))
    print(model)
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(model, next(iter(loader))[0].to(device))
    print(next(iter(loader))[0].size())
    #summary(model, next(iter(train_loader))[0][0].size())
    
    print("Total nb param: ", sum(param.numel() for param in model.parameters()))

    # Get losses, metrics and optimizers
    metric_functions = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
    best_metric = {"mIoU": -1.0}
    
    model.eval()
    
    with torch.no_grad():
        
        # Main loop on validation set
        for (idx, batch_data) in enumerate(tqdm(loader)):
            
            outputs = apply_inference_model(config, batch_data, model, metric_functions, device, idx, writer)
        
        aggregate_metrics_and_save(config, metric_functions, best_metric, model, log_dir, writer)
    
    
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
