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
    
    
def apply_inference_model(config, batch_data, model, metric_functions, device, epoch, idx, cluster, writer):
    
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
            writer.add_figure(f"Validation_prediction_epoch_{epoch}_cluster_{cluster}", fig, global_step=idx*len(inputs)+i)
        
    return prediction


def aggregate_metrics_and_save(config, metric_functions, best_current, model, log_dir, epoch, cluster, writer):
        
    report = metric_functions[0].aggregate()
    metric_functions[0].reset()
    
    pprint(report)
    
    writer.add_scalar(f"Validation/mIoU_cluster_{cluster}", report["mIoU"], epoch + 1)
    for c, iou in report["IoUs"].items():
        writer.add_scalar(f"Validation/IoU_{c}_cluster_{cluster}", iou, epoch + 1)
    
    if best_current["mIoU"] < report["mIoU"]:
        
        best_current.update(report)
        best_current["epoch"] = epoch + 1 
        
        print(f"Saving new best model cluster {cluster}")
        
        with open(os.path.join(log_dir, f'final_results_cluster_{cluster}.json'), 'w') as fp:
            json.dump(best_current, fp, indent=4)
            
        torch.save(model.state_dict(), os.path.join(log_dir, f"best_model_cluster_{cluster}.pth"))
        
    if (epoch + 1) in config["save_epochs"] :
        shutil.copyfile(os.path.join(log_dir, f"best_model_cluster_{cluster}.pth"), os.path.join(log_dir, f"best_model_cluster_{cluster}_epoch_{epoch+1}.pth"))
    
    return report
        

def save_final_metrics(config, best_metric, writer):
    
    # Adding hyperparameters and result values to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
        
    result = {}
    for cluster in best_metric:
        result.update({f"mIoU_cluster_{cluster}": best_metric[cluster]["mIoU"], f"epoch_cluster_{cluster}": best_metric[cluster]["epoch"]})
        
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
    
    # Save the partitioning
    with open(os.path.join(log_dir, 'part_train_samples.json'), 'w') as f:
        json.dump(local_samples, f, indent=4)


    # Build clustered local datasets
    if config["clustering"] == "prior":
        # Perfect split between Cityscapes and GTA for each client
        # Final struct of the datasets and loaders dict: {cluster: {client: ds or loader}}
        train_ds_dict = {"Cityscapes":{}, "GTA":{}} 
        train_loader_dict = {"Cityscapes":{}, "GTA":{}}
        for c in clients:
            if "Cityscapes" in local_samples[c]:
                train_ds_dict["Cityscapes"][c] = torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"])
                train_loader_dict["Cityscapes"][c] = torch.utils.data.DataLoader(train_ds_dict["Cityscapes"][c], batch_size=batch_size, shuffle=True, num_workers=0)
            if "GTA" in local_samples[c]:
                train_ds_dict["GTA"][c] = torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"])
                train_loader_dict["GTA"][c] = torch.utils.data.DataLoader(train_ds_dict["GTA"][c], batch_size=batch_size, shuffle=True, num_workers=0)
 
        # Simply one validation dataset and loader for each cluster, no further partitioning
        val_ds_dict = {"Cityscapes": val_ds_city,
                       "GTA": val_ds_gta}
        val_loader_dict = {"Cityscapes": torch.utils.data.DataLoader(val_ds_dict["Cityscapes"], batch_size=batch_size, 
                                                               shuffle=False, num_workers=0),
                           "GTA": torch.utils.data.DataLoader(val_ds_dict["GTA"], batch_size=batch_size, 
                                                                                  shuffle=False, num_workers=0)}
        example_cluster = "Cityscapes"
        example_client = list(train_ds_dict[example_cluster].keys())[0]
        
    elif config["clustering"] == "computed":
        # Build clustered local train datasets using clustering results 
        # Final structure : train_ds_dict = {cluster: {client: Concat(city, gta)}}
        train_ds_dict = {}
        train_loader_dict = {}
        
        with open(os.path.join(config["clustering_dir"], config["clustered_numbers"]), 'r') as f:
            clustered_numbers = json.load(f)
    
        cluster_institution_numbers = merge_inst_and_cluster_numbers(local_samples, clustered_numbers)
        
        # Merge institutional split and clustering results before creating datasets
        # Final struct of the original dict: {cluster: {client: []}}
        for cluster, cluster_datasets in cluster_institution_numbers.items():
            
            train_ds_dict[cluster] = {}
            train_loader_dict[cluster] = {}
            
            for client, client_cluster_datasets in cluster_datasets.items():
                
                if "Cityscapes" not in client_cluster_datasets:
                    train_ds_dict[cluster][client] = torch.utils.data.Subset(train_ds_gta, client_cluster_datasets["GTA"])
                elif "GTA" not in client_cluster_datasets:
                    train_ds_dict[cluster][client] = torch.utils.data.Subset(train_ds_city, client_cluster_datasets["Cityscapes"])
                else:
                    train_ds_dict[cluster][client] = ConcatDataset([
                        torch.utils.data.Subset(train_ds_city, client_cluster_datasets["Cityscapes"]),
                        torch.utils.data.Subset(train_ds_gta, client_cluster_datasets["GTA"])
                    ])
                    
                train_loader_dict[cluster][client] = torch.utils.data.DataLoader(train_ds_dict[cluster][client], batch_size=batch_size, 
                                                                                 shuffle=True, num_workers=0)
                
        pprint(train_ds_dict)
        
        # Build validation set(s)
        if config["validation_mode"] == "oracle":
            
            val_ds_dict = {}
            val_loader_dict = {}
            
            with open(os.path.join(config["applied_clustering_dir"], config["clustered_numbers_val"]), 'r') as f:
                cluster_institution_numbers_val = json.load(f)
 
            for cluster, cluster_datasets in cluster_institution_numbers_val.items():
                
                if "Cityscapes" not in cluster_datasets:
                    val_ds_dict[cluster] = torch.utils.data.Subset(val_ds_gta, cluster_datasets["GTA"])
                elif "GTA" not in cluster_datasets:
                    val_ds_dict[cluster] = torch.utils.data.Subset(val_ds_city, cluster_datasets["Cityscapes"])
                else:
                    val_ds_dict[cluster] = ConcatDataset([
                        torch.utils.data.Subset(val_ds_city, cluster_datasets["Cityscapes"]),
                        torch.utils.data.Subset(val_ds_gta, cluster_datasets["GTA"])
                    ])
            
                val_loader_dict[cluster] = torch.utils.data.DataLoader(val_ds_dict[cluster], batch_size=batch_size, 
                                                                       shuffle=False, num_workers=0)
            
            example_cluster = list(train_ds_dict.keys())[0]
            example_client = list(train_ds_dict[example_cluster].keys())[0]

    """
    for im, labels in train_ds_dict["0"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["0"]["0"]))
    
    for im, labels in train_ds_dict["1"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["1"]["0"]))
    """
    pprint(train_ds_dict)
    pprint(val_ds_dict)
    im, labels = next(iter(train_loader_dict[example_cluster][example_client]))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
    
    train_sizes = {cluster:np.sum([len(train_ds_dict[cluster][client]) for client in train_ds_dict[cluster]]) for cluster in train_ds_dict}
    val_sizes = {cluster:len(val_ds_dict[cluster]) for cluster in train_ds_dict}
    print("Train sizes: ", train_sizes)
    print("Val sizes: ", val_sizes)
    print("Size of decentralized training set : ", sum([sum([len(train_ds_dict[i][j]) for j in train_ds_dict[i]]) for i in train_ds_dict]))
    print("Size of decentralized validation set : ", np.sum([len(val_ds_dict[i]) for i in val_ds_dict]))
    
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
            cluster_models = {cluster:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for cluster in train_ds_dict}
            local_models = {client:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for client in clients}
            
        elif config["labels"] == "categories":
            cluster_models = {cluster:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for cluster in train_ds_dict}
            local_models = {client:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for client in clients}
      
    for cluster in train_ds_dict:
        cluster_models[cluster].load_state_dict(torch.load(os.path.join(config['previous_train_dir'], config["model_init"])))
        
    split_round = int(config["model_init"][-7:-4])
    num_rounds = config["max_epochs"] - split_round
    if config["lr_scheduler"] == "exp_decay":
        learning_rate = config["learning_rate"]*(config["gamma"]**split_round)
    else:
        learning_rate = config["learning_rate"]
    print(cluster_models[example_cluster])
        
    local_optimizers = {cluster:{client:optim.SGD(local_models[client].parameters(), lr=learning_rate, momentum=config["momentum"]) for client in train_ds_dict[cluster]} for cluster in train_ds_dict}
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(cluster_models[example_cluster], next(iter(train_loader_dict[example_cluster][example_client]))[0].to(device))
    print(next(iter(train_loader_dict[example_cluster][example_client]))[0].size())
    #summary(cluster_models["0"], next(iter(train_loader_dict["0"]["0"]))[0][0].size())
    
    # Get losses, metrics and optimizers
    loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    metric_functions = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
    best_metric = {cluster:{"mIoU": -1.0} for cluster in train_ds_dict}

    if config["lr_scheduler"] == "exp_decay":
        local_schedulers = {cluster:{client:torch.optim.lr_scheduler.ExponentialLR(local_optimizers[cluster][client], config["gamma"], verbose=True) for client in train_ds_dict[cluster]} for cluster in train_ds_dict}
    
    for cluster in train_ds_dict:
        
        cluster_models[cluster].eval()
        
        with torch.no_grad():
            
            # Main loop on validation set
            for (idx, batch_data) in enumerate(tqdm(val_loader_dict[cluster])):
                
                outputs = apply_inference_model(config, batch_data, cluster_models[cluster], metric_functions, device, -1, idx, cluster, writer)
            
            aggregate_metrics_and_save(config, metric_functions, best_metric[cluster], cluster_models[cluster], log_dir, -1, cluster, writer)        
            
    for comm in range(split_round, config["max_epochs"]):  # loop over the dataset multiple times
                
        #barycenter = 
        
        for cluster in train_ds_dict:

            for client in train_ds_dict[cluster]:
                
                for local_param, global_param in zip(local_models[client].parameters(), cluster_models[cluster].parameters()):
                    local_param.data = global_param.data.clone()
                            
                epoch_loss = 0.0
                
                local_models[client].train()
                
                writer.add_scalar("Learning rate/Epoch", local_optimizers[cluster][client].param_groups[0]['lr'], comm+1)
                
                for idx, batch_data in enumerate(tqdm(train_loader_dict[cluster][client])):
        
                    # zero the parameter gradients
                    local_optimizers[cluster][client].zero_grad()
            
                    # forward + backward + optimize
                    outputs, loss, batch_size = apply_train_model(config, batch_data, local_models[client], loss_function, device)
                    
                    loss.backward()
                        
                    local_optimizers[cluster][client].step()
            
                    # print statistics
                    epoch_loss += loss.item()*batch_size
            
                epoch_loss /= len(train_ds_dict[cluster][client])
            
                # Add loss value to tensorboard, and print it
                writer.add_scalar(f"Loss/train/Cluster {cluster}, client {client}", epoch_loss, comm+1)
                print(f"\nCluster {cluster}, client {client}, comm {comm + 1}, average loss: {epoch_loss:.4f}")
            
            if config["lr_scheduler"] == "exp_decay":
                for client in train_ds_dict[cluster]:
                    local_schedulers[cluster][client].step()
               
            for param in cluster_models[cluster].parameters():
                param.data = torch.zeros_like(param.data)
            
            for client in train_ds_dict[cluster]:  
                for global_param, client_param in zip(cluster_models[cluster].parameters(), local_models[client].parameters()):
                    global_param.data += client_param.data.clone() * len(train_ds_dict[cluster][client])/train_sizes[cluster]  
            
            # Validation step
            if (comm) % config["validation_interval"] == 0:
                
                cluster_models[cluster].eval()
                
                with torch.no_grad():
                    
                    # Main loop on validation set
                    for (idx, batch_data) in enumerate(tqdm(val_loader_dict[cluster])):
                        
                        outputs = apply_inference_model(config, batch_data, cluster_models[cluster], metric_functions, device, comm, idx, cluster, writer)
                    
                    aggregate_metrics_and_save(config, metric_functions, best_metric[cluster], cluster_models[cluster], log_dir, comm, cluster, writer)
    
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
