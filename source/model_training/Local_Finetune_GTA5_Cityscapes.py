import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
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

from pprintpp import pprint

#from ..utils.selection_utils import get_model, get_transform, get_loss_metric

from plot_utils import imshow
from torchvision.utils import draw_segmentation_masks
from data.cityscapes_labels import CsLabels
from data.datasets import CacheCityscapes, CacheGTA
from data.transforms import prepare_plot_im, prepare_plot_label, \
    generate_transform_cityscapes_im, generate_transform_cityscapes_label, generate_transform_GTA5_label
from models.models import modified_get_conv_layer
from metrics.metrics import CumulativeClassificationMetrics, CumulativeIoUCityscapes
from data.partitioning_utils import GTA_Cityscapes_iid, GTA_Cityscapes_Full_non_iid, GTA_Cityscapes_Dirichlet_non_iid, assert_GTA_Cityscapes_partitioning
from data.data_utils import recursive_add_parent_in_path, merge_inst_and_cluster_numbers

import psutil
    
    
def apply_train_model(config, batch_data, model, loss_function, device):
    
        inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
        outputs = model(inputs)
        if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
            outputs = resize(outputs, size=labels.shape[-2:])
        loss = loss_function(outputs, labels.squeeze(1))
        
        return outputs, loss, inputs.shape[0] 
    
    
def apply_inference_model(config, batch_data, model, metric_functions, device, epoch, idx, client, writer):
    
    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    outputs = model(inputs)
    if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
        outputs = resize(outputs, size=labels.shape[-2:])
    _, prediction = torch.max(outputs.data, 1, keepdim=True)
    
    for metric_function in metric_functions:
        metric_function(prediction, labels)
        
    if config["print_predictions"] and (epoch + 1) % config["print_validation_interval"] == 0 and idx < 5:
        for i in range(len(inputs)):
            fig, axs = plt.subplots(2, 1)
            print(inputs[i].shape, labels[i].shape, prediction[i].shape)
            axs[0].imshow(prepare_plot_im(inputs[i]))
            axs[0].imshow(prepare_plot_label(labels[i][0], labels=config["labels"]), alpha=0.7)
            axs[1].imshow(prepare_plot_im(inputs[i]))
            axs[1].imshow(prepare_plot_label(prediction[i][0], labels=config["labels"]), alpha=0.7)
            plt.tight_layout()
            writer.add_figure(f"Validation_prediction_epoch_{epoch}_client_{client}", fig, global_step=idx*len(inputs)+i)
        
    return prediction


def aggregate_metrics_and_save(config, metric_functions, best_current, model, log_dir, epoch, client, writer):
        
    report = metric_functions[0].aggregate()
    metric_functions[0].reset()
    
    pprint(report)
    
    writer.add_scalar(f"Validation/mIoU_client_{client}", report["mIoU"], epoch + 1)
    for c, iou in report["IoUs"].items():
        writer.add_scalar(f"Validation/IoU_{c}_client_{client}", iou, epoch + 1)
    
    if best_current["mIoU"] < report["mIoU"]:
        
        best_current.update(report)
        best_current["epoch"] = epoch + 1
        
        print(f"Saving new best model client {client}")
        
        with open(os.path.join(log_dir, f'final_results_client_{client}.json'), 'w') as fp:
            json.dump(best_current, fp, indent=4)
            
        torch.save(model.state_dict(), os.path.join(log_dir, f"best_model_client_{client}.pth"))
        
    if (epoch + 1) in config["save_epochs"] :
        shutil.copyfile(os.path.join(log_dir, f"best_model_client_{client}.pth"), os.path.join(log_dir, f"best_model_client_{client}_epoch_{epoch+1}.pth"))
    
    return report
        

def save_final_metrics(config, best_metric, writer):
    
    # Adding hyperparameters and result values to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
        
    result = {}
    for client in best_metric:
        result.update({f"mIoU_client_{client}": best_metric[client]["mIoU"], f"epoch_client_{client}": best_metric[client]["epoch"]})
        
    writer.add_hparams(config_hparam, result)
        
    

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
    train_transform_city = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_cityscapes_label(size=config["size"], labels=config["labels"]))
    val_transform_city = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_cityscapes_label(size=config["size"], labels=config["labels"]))
    train_transform_gta = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_GTA5_label(size=config["size"], labels=config["labels"]))
    val_transform_gta = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_GTA5_label(size=config["size"], labels=config["labels"]))
    
    # Get train and validation sets, datasets and loaders
    batch_size = config["batch_size"]

    if config["cache"]:
        train_ds_city = CacheCityscapes(config["data_dir_cityscapes"], split='train', mode='fine', target_type='semantic',
                                   cache_transform=train_transform_city[0], cache_target_transform=train_transform_city[1], sort=config["sort"])
        val_ds_city = CacheCityscapes(config["data_dir_cityscapes"], split='val', mode='fine', target_type='semantic', 
                                 cache_transform=val_transform_city[0], cache_target_transform=val_transform_city[1], sort=config["sort"])   
        train_ds_gta = CacheGTA(config["data_dir_gta"], split='train',
                                   cache_transform=train_transform_gta[0], cache_target_transform=train_transform_gta[1], sort=config["sort"])
        val_ds_gta = CacheGTA(config["data_dir_gta"], split='val',
                                 cache_transform=val_transform_gta[0], cache_target_transform=val_transform_gta[1], sort=config["sort"])
    else:
        train_ds_city = CacheCityscapes(config["data_dir_cityscapes"], split='train', mode='fine', target_type='semantic',
                                   transform=train_transform_city[0], target_transform=train_transform_city[1], sort=config["sort"])
        val_ds_city = CacheCityscapes(config["data_dir_cityscapes"], split='val', mode='fine', target_type='semantic', 
                                 transform=val_transform_city[0], target_transform=val_transform_city[1], sort=config["sort"])   
        train_ds_gta = CacheGTA(config["data_dir_gta"], split='train',
                                   transform=train_transform_gta[0], target_transform=train_transform_gta[1], sort=config["sort"])
        val_ds_gta = CacheGTA(config["data_dir_gta"], split='val',
                                 transform=val_transform_gta[0], target_transform=val_transform_gta[1], sort=config["sort"])
    
    print(len(train_ds_city), len(train_ds_city[0]), train_ds_city.__getitem__(0))
    
    # Equilibrate the number of samples from cityscapes and GTA5 in the final training set.
    min_dataset_size_train = min(len(train_ds_city), len(train_ds_gta))
    min_dataset_size_val = min(len(val_ds_city), len(val_ds_gta))
    
    print(min_dataset_size_train, min_dataset_size_val)
    
    if config["cache"]:
        # Preload images and labels in cache to accelerate training
        train_ds_city.cache()
        train_ds_gta.cache()
        val_ds_city.cache()
        val_ds_gta.cache()
        
    for i in range(len(train_ds_city)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(train_ds_city[i][0]))
        axs[1].imshow(prepare_plot_label(train_ds_city[i][1][0], labels=config["labels"]), alpha=1.)
        plt.title("Cityscapes")
        plt.tight_layout()
        writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
        if i > 10:
            break
    for i in range(len(train_ds_gta)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(train_ds_gta[i][0]))
        axs[1].imshow(prepare_plot_label(train_ds_gta[i][1][0], labels=config["labels"]), alpha=1.)
        plt.title("GTA5")
        plt.tight_layout()
        writer.add_figure(f"Verification_GTA", fig, global_step=i)
        if i > 10:
            break
      
    clients = [str(c) for c in list(range(config["nb_clients"]))]
    
    if config["generate_split"] in ["train", "val"]:
        
        if config["generate_split"] == "train":
            size = min_dataset_size_train
        elif config["generate_split"] == "val":
            size = min_dataset_size_val
            
        if config["iid"]:
            if not os.path.exists("../../datasets/Cityscapes_test/IID_partitioning"):
                os.mkdir("../../datasets/Cityscapes_test/IID_partitioning")
            local_samples = GTA_Cityscapes_iid(size, clients, save_folder="../../datasets/Cityscapes_test/IID_partitioning", seed=0, split=config["generate_split"])
            
        else:
            if config["dirichlet"] == 0:
                if not os.path.exists("../../datasets/Cityscapes_test/Full_non_IID_partitioning"):
                    os.mkdir("../../datasets/Cityscapes_test/Full_non_IID_partitioning")
                local_samples = GTA_Cityscapes_Full_non_iid(size, clients, save_folder="../../datasets/Cityscapes_test/Full_non_IID_partitioning", seed=1, split=config["generate_split"])
                
            else:
                # dir 0.25: seed 6 ok, seed 5 less homogeneous but more clustered, seed 20 best so far
                # dir 0.4: seed 20 ok but ùaybe to homogeneous, seed 18 ok
                if not os.path.exists("../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning"):
                    os.mkdir("../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning")
                local_samples = GTA_Cityscapes_Dirichlet_non_iid(size, clients, config["dirichlet"], save_folder="../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning", seed=20, split=config["generate_split"])
    else:
        with open(os.path.join(config["partitioning_dir"], config["train_partition_file"]), 'r') as f:
            local_samples = json.load(f)
        with open(os.path.join(config["partitioning_dir"], config["val_partition_file"]), 'r') as f:
            val_local_samples = json.load(f)

    clients = list(local_samples.keys())
        
    # Assure that we use a correct partitioning of the data
    assert_GTA_Cityscapes_partitioning(local_samples, clients)
    assert_GTA_Cityscapes_partitioning(val_local_samples, clients)
    
    # Save the partitioning
    with open(os.path.join(log_dir, 'part_train_samples.json'), 'w') as f:
        json.dump(local_samples, f, indent=4)
    with open(os.path.join(log_dir, 'part_val_samples.json'), 'w') as f:
        json.dump(val_local_samples, f, indent=4)

    # Isolate local datasets
    local_datasets = {} 
    for c in clients:
        if "Cityscapes" not in local_samples[c]:
            local_datasets[c] = torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"])
        elif "GTA" not in  local_samples[c]:
            local_datasets[c] = torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"])
        else:
            local_datasets[c] = ConcatDataset([
                torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"]),
                torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"])
            ])
        
    local_loaders = {c:torch.utils.data.DataLoader(local_datasets[c], batch_size=batch_size, shuffle=True, num_workers=0) for c in clients}
    
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
    
    """
    for im, labels in train_ds_dict["0"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["0"]["0"]))
    
    for im, labels in train_ds_dict["1"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["1"]["0"]))
    """
    print(local_datasets)
    print(val_local_datasets)
    im, labels = next(iter(local_datasets[example_client]))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    
    print("Train sizes: ", [len(local) for local in local_datasets.values()])
    print("Val sizes: ", [len(local) for local in val_local_datasets.values()])
    print("Size of decentralized training set : ", sum([len(local) for local in local_datasets.values()]))
    print("Size of decentralized validation set : ", np.sum([len(local) for local in val_local_datasets.values()]))
    
    if config["labels"] == "trainId":
        ignore_index = 255
        nb_classes = 20
        ignore_background = True
        
    elif config["labels"] == "categories":
        ignore_index = 255
        nb_classes = 8
        ignore_background = True
        
    device = torch.device("cuda:0")
    
    if config["model"] == "EfficientVIT":
        # Build the model and its parallel counterpart
        if config["labels"] == "trainId":
            local_models = {client:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for client in clients}
            
        elif config["labels"] == "categories":
            local_models = {client:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for client in clients}
      
    # Load each client model with a common well performing initialization
    for client in clients:
        local_models[client].load_state_dict(torch.load(os.path.join(config['previous_train_dir'], config["model_init"])))
        
    split_round = int(config["model_init"][-7:-4])
    num_rounds = config["max_epochs"] - split_round

    print(local_models[example_client])
        
    local_optimizers = {client:optim.SGD(local_models[client].parameters(), lr=config["learning_rate"], momentum=config["momentum"]) for client in clients}
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(local_models[example_client], next(iter(local_loaders[example_client]))[0].to(device))
    print(next(iter(local_loaders[example_client]))[0].size())
    #summary(cluster_models["0"], next(iter(train_loader_dict["0"]["0"]))[0][0].size())
    
    # Get losses, metrics and optimizers
    loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    metric_functions = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
    best_metric = {client:{"mIoU": -1.0} for client in clients}

    for client in clients:
        
        local_models[client].eval()
        
        # First validate the intialization
        with torch.no_grad():
            
            # Main loop on validation set
            for (idx, batch_data) in enumerate(tqdm(val_local_loaders[client])):
                
                outputs = apply_inference_model(config, batch_data, local_models[client], metric_functions, device, split_round-1, idx, client, writer)
            
            aggregate_metrics_and_save(config, metric_functions, best_metric[client], local_models[client], log_dir, split_round-1, client, writer)        
            
        # Then perform some epochs of local finetuning
        for epoch in range(split_round, config["max_epochs"]):  # loop over the dataset multiple times
                        
            epoch_loss = 0.0
            
            local_models[client].train()
            
            writer.add_scalar("Learning rate/Epoch", local_optimizers[client].param_groups[0]['lr'], epoch+1)
            
            for idx, batch_data in enumerate(tqdm(local_loaders[client])):
    
                # zero the parameter gradients
                local_optimizers[client].zero_grad()
        
                # forward + backward + optimize
                outputs, loss, batch_size = apply_train_model(config, batch_data, local_models[client], loss_function, device)
                
                loss.backward()
                    
                local_optimizers[client].step()
        
                # print statistics
                epoch_loss += loss.item()*batch_size
        
            epoch_loss /= len(local_datasets[client])
        
            # Add loss value to tensorboard, and print it
            writer.add_scalar(f"Loss/train/Client {client}", epoch_loss, epoch+1)
            print(f"\nClient {client}, comm {epoch + 1}, average loss: {epoch_loss:.4f}")
           
            # Validation step
            if (epoch) % config["validation_interval"] == 0:
                
                local_models[client].eval()
                
                with torch.no_grad():
                    
                    # Main loop on validation set
                    for (idx, batch_data) in enumerate(tqdm(val_local_loaders[client])):
                        
                        outputs = apply_inference_model(config, batch_data, local_models[client], metric_functions, device, epoch, idx, client, writer)
                    
                    aggregate_metrics_and_save(config, metric_functions, best_metric[client], local_models[client], log_dir, epoch, client, writer)
    
    save_final_metrics(config, best_metric, writer)
    
    
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
