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

    train_images = np.load(os.path.join(config["dataset"], "train", config["images_name"]))
    train_masks = np.load(os.path.join(config["dataset"], "train", "masks.npy"))        
    val_images = np.load(os.path.join(config["dataset"], "val", config["images_name"]))
    val_masks = np.load(os.path.join(config["dataset"], "val", "masks.npy"))
    test_images = np.load(os.path.join(config["dataset"], "test", config["images_name"]))
    test_masks = np.load(os.path.join(config["dataset"], "test", "masks.npy"))   
    
    if config["flip_horizontal"]:
        # Half of the images of the datasets are flipped
        train_images[len(train_images)//2:] = np.flip(train_images[len(train_images)//2:], axis=-1)
        train_masks[len(train_masks)//2:] = np.flip(train_masks[len(train_masks)//2:], axis=-1)
        val_images[len(val_images)//2:] = np.flip(val_images[len(val_images)//2:], axis=-1)
        val_masks[len(val_masks)//2:] = np.flip(val_masks[len(val_masks)//2:], axis=-1)
        test_images[len(test_images)//2:] = np.flip(test_images[len(test_images)//2:], axis=-1)
        test_masks[len(test_masks)//2:] = np.flip(test_masks[len(test_masks)//2:], axis=-1)
        
    if config["flip_vertical"]:
        # Half of the images of the datasets are flipped
        train_images[len(train_images)//2:] = np.flip(train_images[len(train_images)//2:], axis=-2)
        train_masks[len(train_masks)//2:] = np.flip(train_masks[len(train_masks)//2:], axis=-2)
        val_images[len(val_images)//2:] = np.flip(val_images[len(val_images)//2:], axis=-2)
        val_masks[len(val_masks)//2:] = np.flip(val_masks[len(val_masks)//2:], axis=-2)
        test_images[len(test_images)//2:] = np.flip(test_images[len(test_images)//2:], axis=-2)
        test_masks[len(test_masks)//2:] = np.flip(test_masks[len(test_masks)//2:], axis=-2)
        
    if config["flip_90"]:
        train_images = np.pad(train_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        train_masks = np.pad(train_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        val_images = np.pad(val_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        val_masks = np.pad(val_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        test_images = np.pad(test_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        test_masks = np.pad(test_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        
        train_images[len(train_images)//2:] = np.rot90(train_images[len(train_images)//2:], axes=(-2, -1))
        train_masks[len(train_masks)//2:] = np.rot90(train_masks[len(train_masks)//2:], axes=(-2, -1))
        val_images[len(val_images)//2:] = np.rot90(val_images[len(val_images)//2:], axes=(-2, -1))
        val_masks[len(val_masks)//2:] = np.rot90(val_masks[len(val_masks)//2:], axes=(-2, -1))
        test_images[len(test_images)//2:] = np.rot90(test_images[len(test_images)//2:], axes=(-2, -1))
        test_masks[len(test_masks)//2:] = np.rot90(test_masks[len(test_masks)//2:], axes=(-2, -1))
        
    if config["invert"]:
        # Half of the images of the datasets are inverted
        train_images[len(train_images)//2:] = invert_full_dataset(train_images[len(train_images)//2:])
        val_images[len(val_images)//2:] = invert_full_dataset(val_images[len(val_images)//2:])
        test_images[len(test_images)//2:] = invert_full_dataset(test_images[len(test_images)//2:])
         
    print(train_images.shape, train_masks.shape, val_images.shape, val_masks.shape)

    dataset_size = len(train_images)//2
    clients = [str(c) for c in list(range(config["nb_clients"]))]
    
    train_data = TripleMNISTSegmentationDataset(train_images, train_masks)
    val_data = TripleMNISTSegmentationDataset(val_images, val_masks)
    test_data = TripleMNISTSegmentationDataset(test_images, test_masks)
    
    if config["generate_partitioning"]:
        if config["iid"]:
            if not os.path.exists(os.path.join(config["dataset"], "IID_partitioning")):
                os.mkdir(os.path.join(config["dataset"], "IID_partitioning"))
            local_samples = Inverted_Triple_MNIST_iid(dataset_size, clients, save_folder=os.path.join(config["dataset"], "IID_partitioning"), seed=0)
            
        else:
            if config["fraction"] is not None:
                if not os.path.exists(os.path.join(config["dataset"], f"Fraction_non_IID_partitioning_{config['fraction']}")):
                    os.mkdir(os.path.join(config["dataset"], f"Fraction_non_IID_partitioning_{config['fraction']}"))
                local_samples = Inverted_Triple_MNIST_Fraction_non_iid(dataset_size, clients, config["fraction"], save_folder=os.path.join(config["dataset"], f"Fraction_non_IID_partitioning_{config['fraction']}"), seed=1)
                
            elif config["dirichlet"] == 0:
                if not os.path.exists(os.path.join(config["dataset"], "Full_non_IID_partitioning")):
                    os.mkdir(os.path.join(config["dataset"], "Full_non_IID_partitioning"))
                local_samples = Inverted_Triple_MNIST_Full_non_iid(dataset_size, clients, save_folder=os.path.join(config["dataset"], "Full_non_IID_partitioning"), seed=1)
                
            else:
                # dir 0.25: seed 6 ok, seed 5 less homogeneous but more clustered, seed 20 best so far
                # dir 0.4: seed 20 ok but ùaybe to homogeneous, seed 18 ok
                if not os.path.exists(os.path.join(config["dataset"], "Dirichlet_non_IID_partitioning")):
                    os.mkdir(os.path.join(config["dataset"], "Dirichlet_non_IID_partitioning"))
                local_samples = Inverted_Triple_MNIST_Dirichlet_non_iid(dataset_size, clients, config["dirichlet"], save_folder=os.path.join(config["dataset"], "Dirichlet_non_IID_partitioning"), seed=20)
    else:
        with open(os.path.join(config["partitioning_file"]), 'r') as f:
            local_samples = json.load(f)
        
    clients = list(local_samples.keys())
    
    # Assure that we use a correct partitioning of the data
    assert_Inverted_Triple_MNIST_partitioning(local_samples, clients)

    if config["clustering_per_institution"]:
        
        local_datasets = {} 
        for c in clients:
            full_local_samples = []
            if "Clear" in local_samples[c]:
                full_local_samples.extend(local_samples[c]["Clear"])
            if "Inverted" in  local_samples[c]:
                full_local_samples.extend([ids+dataset_size for ids in local_samples[c]["Inverted"]])
            local_datasets[c] = torch.utils.data.Subset(train_data, full_local_samples)
            
        local_cluster_datasets = {"Clear":{}, "Inverted":{}}
        local_cluster_loaders = {"Clear":{}, "Inverted":{}}
        
        for cluster, client_list in config["institution_clustering"].items():
            for client_id in client_list:
                local_cluster_datasets[cluster][client_id] = local_datasets[client_id]
                local_cluster_loaders[cluster][client_id] = torch.utils.data.DataLoader(local_cluster_datasets[cluster][client_id], batch_size=config["batch_size"], shuffle=True, num_workers=0)
        
        val_cluster_loaders = {"Clear":torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0),
                              "Inverted":torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)}
        
        test_cluster_loaders = {"Clear":torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0),
                              "Inverted":torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)}
    
    else:
        local_cluster_datasets = {"Clear":{}, "Inverted":{}}
        local_cluster_loaders = {"Clear":{}, "Inverted":{}}
        for c in clients:
            if "Clear" in local_samples[c]:
                local_cluster_datasets["Clear"][c] = torch.utils.data.Subset(train_data, local_samples[c]["Clear"])
                local_cluster_loaders["Clear"][c] = torch.utils.data.DataLoader(local_cluster_datasets["Clear"][c], batch_size=config["batch_size"], shuffle=True, num_workers=0)
            if "Inverted" in  local_samples[c]:
                local_cluster_datasets["Inverted"][c] = torch.utils.data.Subset(train_data, [ids+dataset_size for ids in local_samples[c]["Inverted"]])
                local_cluster_loaders["Inverted"][c] = torch.utils.data.DataLoader(local_cluster_datasets["Inverted"][c], batch_size=config["batch_size"], shuffle=True, num_workers=0)  
    
        val_cluster_loaders = {"Clear":torch.utils.data.DataLoader(torch.utils.data.Subset(val_data, list(range(len(val_data)//2))), batch_size=config["batch_size"], shuffle=False, num_workers=0),
                              "Inverted":torch.utils.data.DataLoader(torch.utils.data.Subset(val_data, list(range(len(val_data)//2, len(val_data)))), batch_size=config["batch_size"], shuffle=False, num_workers=0)}
        
        test_cluster_loaders = {"Clear":torch.utils.data.DataLoader(torch.utils.data.Subset(test_data, list(range(len(test_data)//2))), batch_size=config["batch_size"], shuffle=False, num_workers=0),
                              "Inverted":torch.utils.data.DataLoader(torch.utils.data.Subset(test_data, list(range(len(test_data)//2, len(test_data)))), batch_size=config["batch_size"], shuffle=False, num_workers=0)}
    
    if len(config["digits"]) > 0:
        classes = ["Background"]+[str(d) for d in config["digits"]]
        nb_classes = len(config["digits"])+1
    else:
        classes = ["Background"]+list("0123456789")
        nb_classes = 11
        
    print(train_data)
    print(val_data)
    print(len(train_data))
    print(len(val_data))
    plt.imshow(train_data[0][0][0], cmap="gray")
    plt.title(train_data[0][1])
    plt.show()
    plt.imshow(train_data[len(train_data)//2+100][0][0], cmap="gray")
    plt.title(train_data[len(train_data)//2+100][1])
    plt.show()
    
    print(train_data[len(train_data)//2+100][0].min(), train_data[len(train_data)//2+100][0].max())
    print(train_data[0][0].min(), train_data[0][0].max())
    
    best_val_perfs = []
    test_perfs = []
    
    for restart in range(config["nb_restart"]):

        local_models = {c:FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device) for c in clients}
        
        cluster_models = {"Clear":FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device), "Inverted":FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device)}
        clusters = ["Clear", "Inverted"]
        
        print(cluster_models[clusters[0]])
        summary(cluster_models[clusters[0]], (1,64,96))
        
        weights = torch.tensor([0.01] + [0.99/(nb_classes-1)]*(nb_classes-1)).to(device)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        
        metric = CumulativeIoUTripleMNISTSegmentation(num_classes=nb_classes, classes=classes)
        
        # Cluster finetuning
        # Load best validation model before finetuning
        for cluster in clusters:
            if config["split_round"] == 0:
                cluster_models[cluster].load_state_dict(torch.load(os.path.join(config["model_dir"], f"model_restart_{restart}.pth")))
            else:
                cluster_models[cluster].load_state_dict(torch.load(os.path.join(config["model_dir"], f"model_restart_{restart}_round_{config['split_round']}.pth")))
        
        num_rounds = config["max_rounds"] - config["split_round"]
        if config["lr_scheduler"] == "exp_decay":
            learning_rate = config["learning_rate"]*(config["gamma"]**config["split_round"])
        else:
            learning_rate = config["learning_rate"]
            
        local_optimizers = {cluster:{client:optim.SGD(local_models[client].parameters(), lr=learning_rate, momentum=config["momentum"]) for client in local_cluster_datasets[cluster]} for cluster in local_cluster_datasets}
        
        best_metric = {cluster:0 for cluster in clusters}
        
        if config["lr_scheduler"] == "exp_decay":
            local_schedulers = {cluster:{client:torch.optim.lr_scheduler.ExponentialLR(local_optimizers[cluster][client], config["gamma"], verbose=True) for client in local_cluster_datasets[cluster]} for cluster in local_cluster_datasets}

        for comm in range(config["split_round"], config["max_rounds"]):  # loop over the dataset multiple times
        
            total_cluster_pred = []
            total_cluster_truth = []
            
            for idc, cluster in enumerate(clusters):
                
                for idcl, client in enumerate(local_cluster_datasets[cluster]):
                
                    for local_param, cluster_param in zip(local_models[client].parameters(), cluster_models[cluster].parameters()):
                        local_param.data = cluster_param.data.clone()
                        
                    local_models[client].train()
                    
                    # Train the model
                    epoch_loss = 0.0
                    
                    for i, (images, labels) in enumerate(tqdm(local_cluster_loaders[cluster][client])):
            
                        output = local_models[client](images.to(device))     
                        loss = loss_func(output, labels.to(device))
                        
                        # clear gradients for this training step   
                        local_optimizers[cluster][client].zero_grad()
                        
                        # backpropagation, compute gradients 
                        loss.backward()    
                        
                        # apply gradients             
                        local_optimizers[cluster][client].step()                
                        epoch_loss += loss.item()*len(images)
                        
                        writer.add_scalar(f"Loss/train/Restart_{restart}_Cluster_{cluster}_Client_{client}_per_iteration", loss.item(), comm*len(local_cluster_loaders[cluster][client])+i+1)
                        
                    epoch_loss /= len(local_cluster_datasets[cluster][client])    
                    
                    print ('\nCluster {}, Client [{}/{}], Comm [{},{}], Loss: {:.4f}'.format(cluster, idcl, len(local_cluster_datasets[cluster]), comm + 1, config["max_rounds"], epoch_loss))
                    writer.add_scalar(f"Loss/train/Restart_{restart}_Client_{client}_cluster_{cluster}", epoch_loss, comm+1)
                
                # Très gros con juste en dessous 
                # assert np.all([torch.equal(init, param.data) for init, param in zip(initial_state.values(), model.parameters())])
                if config["lr_scheduler"] == "exp_decay":
                    for client in local_cluster_datasets[cluster]:
                        local_schedulers[cluster][client].step()
                        
                for param in cluster_models[cluster].parameters():
                    param.data = torch.zeros_like(param.data)
                
                for client in local_cluster_datasets[cluster]:  
                    for cluster_param, client_param in zip(cluster_models[cluster].parameters(), local_models[client].parameters()):
                        cluster_param.data += client_param.data.clone() * len(local_cluster_datasets[cluster][client])/(len(train_data)//2) 
                        
                # Test the model
                cluster_models[cluster].eval()    
                
                with torch.no_grad():
    
                    for idl, (images, labels) in enumerate(tqdm(val_cluster_loaders[cluster])):
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        test_output = cluster_models[cluster](images)
                        _, pred_y = torch.max(test_output, 1, keepdim=True)

                        metric(y_pred=pred_y, y=labels)
                            
                    report = metric.aggregate()
                    metric.reset()

                    print(report)

                    for cl in classes:
                        writer.add_scalar(f"Validation/Restart_{restart}_Cluster_{cluster}_IoU_{cl}", report["IoUs"][cl], comm + 1)
                    writer.add_scalar(f"Validation/Restart_{restart}_Cluster_{cluster}_mIoU", report["mIoU"], comm + 1)
                    
                    if report["mIoU"] > best_metric[cluster]:
                        best_metric[cluster] = report["mIoU"]
                        print("Saving new best model")
                        torch.save(cluster_models[cluster].state_dict(), os.path.join(log_dir, f"model_cluster_{cluster}_restart_{restart}.pth"))
            
                        with open(os.path.join(log_dir, f'val_perf_cluster_{cluster}_restart_{restart}.json'), 'w') as f:
                            json.dump(report, f, indent=4) 
        
        # Total best validation
        total_cluster_pred = []
        total_cluster_truth = []
        
        for cluster in clusters:
            
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(log_dir, f"model_cluster_{cluster}_restart_{restart}.pth")))
        
            # Test the model
            cluster_models[cluster].eval()    
            
            with torch.no_grad():

                for idl, (images, labels) in enumerate(tqdm(val_cluster_loaders[cluster])):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    test_output = cluster_models[cluster](images)
                    _, pred_y = torch.max(test_output, 1, keepdim=True)

                    metric(y_pred=pred_y, y=labels)
           
        report = metric.aggregate()
        metric.reset()
        best_val_perfs.append(report["mIoU"])
        
        # Final test performance
        total_cluster_pred = []
        total_cluster_truth = []
        
        for cluster in clusters:
            
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(log_dir, f"model_cluster_{cluster}_restart_{restart}.pth")))
        
            # Test the model
            cluster_models[cluster].eval()    
            
            with torch.no_grad():

                for idl, (images, labels) in enumerate(tqdm(test_cluster_loaders[cluster])):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    test_output = cluster_models[cluster](images)
                    _, pred_y = torch.max(test_output, 1, keepdim=True)

                    metric(y_pred=pred_y, y=labels)
                        
         
        report = metric.aggregate()
        metric.reset()
        test_perfs.append(report["mIoU"])
    
    df_metric = pd.DataFrame(data={"validation_mIoU":best_val_perfs, "test_mIoU":test_perfs})
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