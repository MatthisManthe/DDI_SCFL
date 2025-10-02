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
    
    
def apply_train_model(config, batch_data, model, loss_function, device):
    
        inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
        outputs = model(inputs)
        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = resize(outputs, size=labels.shape[-2:])
        loss = loss_function(outputs, labels.squeeze(1))
        
        return outputs, loss, inputs.shape[0] 
    
    
def apply_inference_model(config, batch_data, model, metric_functions, device, epoch, idx, writer):
    
    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    outputs = model(inputs)
    if outputs.shape[-2:] != labels.shape[-2:]:
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
    batch_size = config["nb_gpus"]*config["batch_size_per_gpu"]

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
    print(len(train_ds_gta), len(train_ds_gta[0]), train_ds_gta.__getitem__(0))
    
    # Equilibrate the number of samples from cityscapes and GTA5 in the final training set.
    min_dataset_size_train = min(len(train_ds_city), len(train_ds_gta))
    min_dataset_size_val = min(len(val_ds_city), len(val_ds_gta))
    
    print(min_dataset_size_train, min_dataset_size_val)
        
    # Cache and build the final training and validation sets with loaders
    if config["datasets"] == "both":
        
        if config["cache"]:
            train_ds_city.cache()
            train_ds_gta.cache()
            val_ds_city.cache()
            val_ds_gta.cache()
            
        train_data = ConcatDataset([train_ds_city, train_ds_gta])
        val_data = ConcatDataset([val_ds_city, val_ds_gta])
        
    elif config["datasets"] == "GTA":
        
        if config["cache"]:
            train_ds_gta.cache()
            val_ds_gta.cache()
            
        train_data = train_ds_gta
        val_data = val_ds_gta
        
    elif config["datasets"] == "Cityscapes":
        
        if config["cache"]:
            train_ds_city.cache()
            val_ds_city.cache()
            
        train_data = train_ds_city
        val_data = val_ds_city
        
    else:
        raise Exception("Choose the right dataset(s) bro.") 
                
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                              shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, 
                                              shuffle=False, num_workers=0)
    
    if config["datasets"] in ["both", "Cityscapes"]:
        for i in range(len(train_ds_city)):
            fig, axs = plt.subplots(1, 2, figsize=[12,12])
            axs[0].imshow(prepare_plot_im(train_ds_city[i][0]))
            axs[1].imshow(prepare_plot_label(train_ds_city[i][1][0], labels=config["labels"]), alpha=1.)
            plt.title(f"Cityscapes {train_ds_city.image_paths[i]}")
            plt.tight_layout()
            writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
            if i > 10:
                break
            
            im, labels = train_ds_city[0]
            print("\nCityscapes", im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
            print(len(train_ds_city))
            
    if config["datasets"] in ["both", "GTA"]:
        for i in range(len(train_ds_gta)):
            fig, axs = plt.subplots(1, 2, figsize=[12,12])
            axs[0].imshow(prepare_plot_im(train_ds_gta[i][0]))
            axs[1].imshow(prepare_plot_label(train_ds_gta[i][1][0], labels=config["labels"]), alpha=1.)
            plt.title(f"GTA5 {train_ds_gta.image_paths[i]}")
            plt.tight_layout()
            writer.add_figure(f"Verification_GTA", fig, global_step=i)
            if i > 10:
                break
            
            im, labels = train_ds_gta[0]
            print("\nGTA", im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
            print(len(train_ds_gta))   
    
    im, labels = next(iter(train_loader))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    print(len(train_data), len(val_data))
    
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
        model = seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device)
    elif config["labels"] == "categories":
        model = seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device)
    
    print(model)
    
    if config["nb_gpus"] == 1:
        model_parallel = model
    else:
        model_parallel = torch.nn.DataParallel(model)
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(model, next(iter(train_loader))[0].to(device))
    print(next(iter(train_loader))[0].size())
    #summary(model, next(iter(train_loader))[0][0].size())
    
    print("Total nb param: ", sum(param.numel() for param in model.parameters()))

    # Get losses, metrics and optimizers
    loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    metric_functions = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
    best_metric = {"mIoU": -1.0}
    
    if config["optim"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    elif config["optim"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    if config["lr_scheduler"] == "exp_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config["gamma"], verbose=True)
    
    # Define max number of training epoch
    max_epochs = config["max_epochs"]
    
    for epoch in range(max_epochs):  # loop over the dataset multiple times
    
        epoch_loss = 0.0
        
        model.train()
        
        writer.add_scalar("Learning rate/Epoch", optimizer.param_groups[0]['lr'], epoch+1)
        
        for idx, batch_data in enumerate(tqdm(train_loader)):

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs, loss, batch_size = apply_train_model(config, batch_data, model_parallel, loss_function, device)
            
            loss.backward()
                
            optimizer.step()
    
            # print statistics
            epoch_loss += loss.item()*batch_size
        
        epoch_loss /= len(train_data)
        
        # Add loss value to tensorboard, and print it
        writer.add_scalar("Loss/train", epoch_loss, epoch+1)
        print(f"\nepoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if config["lr_scheduler"] == "exp_decay":
            scheduler.step()
            
        # Validation step
        if (epoch) % config["validation_interval"] == 0:
            
            model.eval()
            
            with torch.no_grad():
                
                # Main loop on validation set
                for (idx, batch_data) in enumerate(tqdm(val_loader)):
                    
                    outputs = apply_inference_model(config, batch_data, model_parallel, metric_functions, device, epoch, idx, writer)
                
                aggregate_metrics_and_save(config, metric_functions, best_metric, model_parallel, log_dir, epoch, writer)
    
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
