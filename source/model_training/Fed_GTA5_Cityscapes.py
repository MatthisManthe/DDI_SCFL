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

import psutil
    
    
def apply_train_model(config, batch_data, model, loss_function, device):
    
        inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
        outputs = model(inputs)
        if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
            outputs = resize(outputs, size=labels.shape[-2:])
        loss = loss_function(outputs, labels.squeeze(1))
        
        return outputs, loss, inputs.shape[0] 
    
    
def apply_inference_model(config, batch_data, model, metric_functions, device, epoch, idx, writer):
    
    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    outputs = model(inputs)
    if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
        outputs = resize(outputs, size=labels.shape[-2:])
    _, prediction = torch.max(outputs.data, 1, keepdim=True)
    
    for metric_function in metric_functions:
        metric_function(prediction, labels)
        
    if config["print_predictions"] and (epoch + 1) % config["print_validation_interval"] == 0 and (idx < 5 or (idx > 500 and idx < 505)):
        for i in range(len(inputs)):
            fig, axs = plt.subplots(2, 1)
            print(inputs[i].shape, labels[i].shape, prediction[i].shape)
            axs[0].imshow(prepare_plot_im(inputs[i]))
            axs[0].imshow(prepare_plot_label(labels[i][0], labels=config["labels"]), alpha=0.7)
            axs[1].imshow(prepare_plot_im(inputs[i]))
            axs[1].imshow(prepare_plot_label(prediction[i][0], labels=config["labels"]), alpha=0.7)
            plt.tight_layout()
            writer.add_figure(f"Validation_prediction_epoch_{epoch}", fig, global_step=idx*len(inputs)+i)
        
    return prediction


def aggregate_metrics_and_save(config, metric_functions, best_current, model, log_dir, epoch, writer):
        
    report = metric_functions[0].aggregate()
    metric_functions[0].reset()
    
    pprint(report)
    
    writer.add_scalar("Validation/mIoU", report["mIoU"], epoch + 1)
    for c, iou in report["IoUs"].items():
        writer.add_scalar(f"Validation/IoU_{c}", iou, epoch + 1)
    
    if best_current["mIoU"] < report["mIoU"]:
        
        best_current.update(report)
        best_current["epoch"] = epoch + 1 
        
        print("Saving new best model")
        
        with open(os.path.join(log_dir,'final_results.json'), 'w') as fp:
            json.dump(best_current, fp, indent=4)
            
        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
        
    if (epoch + 1) in config["save_epochs"] :
        shutil.copyfile(os.path.join(log_dir, "best_model.pth"), os.path.join(log_dir, f"best_model_epoch_{epoch+1}.pth"))
    
    return report
        

def save_final_metrics(config, best_metric, writer):
    
    # Adding hyperparameters and result values to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value

        
    writer.add_hparams(config_hparam, {"mIoU": best_metric["mIoU"], "epoch": best_metric["epoch"]})
        
    

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
       
    # Build the final training and validation sets with loaders
    train_data = ConcatDataset([train_ds_city, train_ds_gta])
    val_data = ConcatDataset([val_ds_city, val_ds_gta])
                
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                              shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, 
                                              shuffle=False, num_workers=0)
    
    im, labels = train_ds_city[0]
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_city))
    
    im, labels = train_ds_gta[0]
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_gta))
    
    im, labels = next(iter(train_loader))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(train_data), len(val_data))
    
    """
    for im, labels in train_ds_city:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_city))
    
    for im, labels in train_ds_gta:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_gta))
    """
    
    if config["labels"] == "trainId":
        ignore_index = 255
        nb_classes = 20
        ignore_background = True
        
    elif config["labels"] == "categories":
        ignore_index = 255
        nb_classes = 8
        ignore_background = True
        
    device = torch.device("cuda:0")

    num_rounds = config["max_epochs"]
    clients = [str(c) for c in list(range(config["nb_clients"]))]

    if config["model"] == "EfficientVIT":
        # Build the model and its parallel counterpart
        if config["labels"] == "trainId":
            model = seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device)
            local_models = {c:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for c in clients}
            
        elif config["labels"] == "categories":
            model = seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device)
            local_models = {c:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for c in clients}
        
    print(model)   
    
    local_optimizers = {c:optim.SGD(local_models[c].parameters(), lr = config["learning_rate"], momentum=config["momentum"]) for c in clients}
    
    if config["generate_partitioning"]:
        if config["iid"]:
            if not os.path.exists("../../datasets/Cityscapes_test/IID_partitioning"):
                os.mkdir("../../datasets/Cityscapes_test/IID_partitioning")
            local_samples = GTA_Cityscapes_iid(min_dataset_size_train, clients, save_folder="../../datasets/Cityscapes_test/IID_partitioning", seed=0)
            
        else:
            if config["dirichlet"] == 0:
                if not os.path.exists("../../datasets/Cityscapes_test/Full_non_IID_partitioning"):
                    os.mkdir("../../datasets/Cityscapes_test/Full_non_IID_partitioning")
                local_samples = GTA_Cityscapes_Full_non_iid(min_dataset_size_train, clients, save_folder="../../datasets/Cityscapes_test/Full_non_IID_partitioning", seed=1)
                
            else:
                # dir 0.25: seed 6 ok, seed 5 less homogeneous but more clustered, seed 20 best so far
                # dir 0.4: seed 20 ok but ùaybe to homogeneous, seed 18 ok
                if not os.path.exists("../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning"):
                    os.mkdir("../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning")
                local_samples = GTA_Cityscapes_Dirichlet_non_iid(min_dataset_size_train, clients, config["dirichlet"], save_folder="../../datasets/Cityscapes_test/Dirichlet_non_IID_partitioning", seed=20)
    else:
        with open(os.path.join(config["partitioning_file"]), 'r') as f:
            local_samples = json.load(f)
        
    clients = list(local_samples.keys())
    
    # Assure that we use a correct partitioning of the data
    assert_GTA_Cityscapes_partitioning(local_samples, clients)
    
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
    
    # Save the partitioning
    with open(os.path.join(log_dir, 'part_train_samples.json'), 'w') as f:
        json.dump(local_samples, f, indent=4)
          
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(model, next(iter(train_loader))[0].to(device))
    print(next(iter(train_loader))[0].size())
    if config["model"] == "UNet":
        summary(model, next(iter(train_loader))[0][0].size())
    
    # Get losses, metrics and optimizers
    loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    metric_functions = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
    best_metric = {"mIoU": -1.0}

    if config["lr_scheduler"] == "exp_decay":
        local_schedulers = {c:torch.optim.lr_scheduler.ExponentialLR(local_optimizers[c], config["gamma"], verbose=True) for c in clients}
    
    for comm in range(num_rounds):  # loop over the dataset multiple times
    
        for c in clients:
            
            for local_param, global_param in zip(local_models[c].parameters(), model.parameters()):
                local_param.data = global_param.data.clone()
                        
            epoch_loss = 0.0
            
            local_models[c].train()
            
            writer.add_scalar("Learning rate/Epoch", local_optimizers[c].param_groups[0]['lr'], comm+1)
            
            for idx, batch_data in enumerate(tqdm(local_loaders[c])):
    
                # zero the parameter gradients
                local_optimizers[c].zero_grad()
        
                # forward + backward + optimize
                outputs, loss, batch_size = apply_train_model(config, batch_data, local_models[c], loss_function, device)
                
                loss.backward()
                    
                local_optimizers[c].step()
        
                # print statistics
                epoch_loss += loss.item()*batch_size
        
            epoch_loss /= len(local_datasets[c])
        
            # Add loss value to tensorboard, and print it
            writer.add_scalar(f"Loss/train/Client {c}", epoch_loss, comm+1)
            print(f"\nClient {c}, comm {comm + 1}, average loss: {epoch_loss:.4f}")
        
        if config["lr_scheduler"] == "exp_decay":
            for c in clients:
                local_schedulers[c].step()
           
        for param in model.parameters():
            param.data = torch.zeros_like(param.data)
        
        for c in clients:  
            for global_param, client_param in zip(model.parameters(), local_models[c].parameters()):
                global_param.data += client_param.data.clone() * len(local_datasets[c])/len(train_data)    
        
        # Validation step
        if (comm) % config["validation_interval"] == 0:
            
            model.eval()
            
            with torch.no_grad():
                
                # Main loop on validation set
                for (idx, batch_data) in enumerate(tqdm(val_loader)):
                    
                    outputs = apply_inference_model(config, batch_data, model, metric_functions, device, comm, idx, writer)
                
                aggregate_metrics_and_save(config, metric_functions, best_metric, model, log_dir, comm, writer)
    
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
