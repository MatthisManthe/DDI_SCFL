import torch
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import copy
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import shutil
import copy
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from torchsummary import summary
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
from torch import optim
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
import pickle 

from sklearn.metrics import classification_report, multilabel_confusion_matrix
from data.datasets import TripleMNISTSegmentationDataset
from data.partitioning_utils import Inverted_Triple_MNIST_iid, Inverted_Triple_MNIST_Full_non_iid, Inverted_Triple_MNIST_Dirichlet_non_iid, \
    Inverted_Triple_MNIST_Fraction_non_iid, assert_Inverted_Triple_MNIST_partitioning
from models.models import FCN_Triple_MNIST_Segmentation
from metrics.metrics import CumulativeIoUTripleMNISTSegmentation
    
def flatten(params):
    return torch.cat([param.flatten() for param in params])


def invert_full_dataset(dataset):
    return 255 - dataset


def main(args, config):
    """Main function"""
    
    # Initializing tensorboard summary
    log_dir = config["log_dir"] + "/" + os.path.basename(__file__)
    if config["log_filename"] is not None:
        log_dir += "/"+config["log_filename"]
    else:
        log_dir += "/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(log_dir=log_dir)
    
    # Copy config file in the experiment folder
    shutil.copy(args.config_path, log_dir)
    device = config["device"]

    test_images = np.load(os.path.join(config["dataset"], "test", config["images_name"]))
    test_masks = np.load(os.path.join(config["dataset"], "test", "masks.npy"))   
    
    if config["flip_horizontal"]:
        # Half of the images of the datasets are flipped
        test_images[len(test_images)//2:] = np.flip(test_images[len(test_images)//2:], axis=-1)
        test_masks[len(test_masks)//2:] = np.flip(test_masks[len(test_masks)//2:], axis=-1)
        
    if config["flip_vertical"]:
        # Half of the images of the datasets are flipped
        test_images[len(test_images)//2:] = np.flip(test_images[len(test_images)//2:], axis=-2)
        test_masks[len(test_masks)//2:] = np.flip(test_masks[len(test_masks)//2:], axis=-2)
        
    if config["flip_90"]:
        test_images = np.pad(test_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        test_masks = np.pad(test_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        
        test_images[len(test_images)//2:] = np.rot90(test_images[len(test_images)//2:], axes=(-2, -1))
        test_masks[len(test_masks)//2:] = np.rot90(test_masks[len(test_masks)//2:], axes=(-2, -1))
        
    if config["invert"]:
        # Half of the images of the datasets are inverted
        test_images[len(test_images)//2:] = invert_full_dataset(test_images[len(test_images)//2:])
         
    print(test_images.shape, test_masks.shape)

    dataset_size = len(test_images)//2
    clients = [str(c) for c in list(range(config["nb_clients"]))]

    test_data = TripleMNISTSegmentationDataset(test_images, test_masks)

    with open(os.path.join(config["partitioning_file"]), 'r') as f:
        local_samples = json.load(f)
        
    clients = list(local_samples.keys())
    
    # Assure that we use a correct partitioning of the data
    assert_Inverted_Triple_MNIST_partitioning(local_samples, clients)

    clusters = ["Clear", "Inverted"]
    
    local_datasets = {} 
    local_cluster_datasets = {"Clear":[], "Inverted":[]}
    for idc, c in enumerate(clients):
        full_local_samples = []
        if "Clear" in local_samples[c]:
            full_local_samples.extend(local_samples[c]["Clear"])
        if "Inverted" in  local_samples[c]:
            full_local_samples.extend([ids+dataset_size for ids in local_samples[c]["Inverted"]])
        local_datasets[c] = torch.utils.data.Subset(test_data, full_local_samples)
        
        if idc < 5:
            local_cluster_datasets["Clear"].append(local_datasets[c])
        else:
            local_cluster_datasets["Inverted"].append(local_datasets[c])
            
    local_cluster_datasets["Clear"] = ConcatDataset(local_cluster_datasets["Clear"])
    local_cluster_datasets["Inverted"] = ConcatDataset(local_cluster_datasets["Inverted"])
    
    loaders = {cluster:torch.utils.data.DataLoader(local_cluster_datasets[cluster], batch_size=config["batch_size"], shuffle=False, num_workers=0) for cluster in clusters}
        
    if len(config["digits"]) > 0:
        classes = ["Background"]+[str(d) for d in config["digits"]]
        nb_classes = len(config["digits"])+1
    else:
        classes = ["Background"]+list("0123456789")
        nb_classes = 11
        
    print(test_data)
    print(len(test_data))
    plt.imshow(test_data[0][0][0], cmap="gray")
    plt.title(test_data[0][1])
    plt.show()
    plt.imshow(test_data[len(test_data)//2+100][0][0], cmap="gray")
    plt.title(test_data[len(test_data)//2+100][1])
    plt.show()
    
    print(test_data[len(test_data)//2+100][0].min(), test_data[len(test_data)//2+100][0].max())
    print(test_data[0][0].min(), test_data[0][0].max())
    
    test_perfs = []
    
    for restart in range(config["nb_restart"]):

        cluster_models = {"Clear":FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device), "Inverted":FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device)}
        
        
        print(cluster_models[clusters[0]])
        summary(cluster_models[clusters[0]], (1,64,96))
        
        weights = torch.tensor([0.01] + [0.99/(nb_classes-1)]*(nb_classes-1)).to(device)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        
        metric = CumulativeIoUTripleMNISTSegmentation(num_classes=nb_classes, classes=classes)
        
        # Cluster finetuning
        # Load best validation model before finetuning
        for cluster in clusters:
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(config["model_dir"], f"model_cluster_{cluster}_restart_{restart}.pth")))

            # Test the model
            cluster_models[cluster].eval()    

            with torch.no_grad():

                for idl, (images, labels) in enumerate(tqdm(loaders[cluster])):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    test_output = cluster_models[cluster](images)
                    _, pred_y = torch.max(test_output, 1, keepdim=True)

                    metric(y_pred=pred_y, y=labels)
           
        report = metric.aggregate()
        metric.reset()
        test_perfs.append(report["mIoU"])
    
    df_metric = pd.DataFrame(data={"test_mIoU":test_perfs})
    df_metric.to_csv(os.path.join(log_dir, "Final_results.csv"))
    

if __name__ == "__main__":
    
    print("Curent working directory: ", os.getcwd())
    
    print("Is cuda avaiable? ", torch.cuda.is_available())
    print("Number of cuda devices available: ", torch.cuda.device_count())
    
    # Define argument parser and its attributes
    parser = argparse.ArgumentParser(description='Train 3D UNet on Brats')
    
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='path to json config file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    main(args, config)