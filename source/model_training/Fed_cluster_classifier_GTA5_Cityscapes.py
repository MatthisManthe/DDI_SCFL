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

from sklearn.metrics import classification_report
from plot_utils import imshow
from torchvision.utils import draw_segmentation_masks
from data.datasets import CacheCityscapesClassification, CacheGTAClassification
from data.transforms import prepare_plot_im, generate_transform_cityscapes_im
from metrics.metrics import CumulativeClassificationMetrics
from data.partitioning_utils import GTA_Cityscapes_iid, GTA_Cityscapes_Full_non_iid, GTA_Cityscapes_Dirichlet_non_iid, assert_GTA_Cityscapes_partitioning
from data.data_utils import recursive_add_parent_in_path, merge_inst_and_cluster_numbers
from models.models import CNN_GTA_Cityscapes_source_Classification

import psutil

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
    transform = generate_transform_cityscapes_im(size=config["size"])
    
    # Get train and validation sets, datasets and loaders
    batch_size = config["batch_size"]

    if config["cache"]:
        train_ds_city = CacheCityscapesClassification(config["data_dir_cityscapes"], split='train', cache_transform=transform, sort=config["sort"])
        val_ds_city = CacheCityscapesClassification(config["data_dir_cityscapes"], split='val', cache_transform=transform, sort=config["sort"])   
        train_ds_gta = CacheGTAClassification(config["data_dir_gta"], split='train', cache_transform=transform, sort=config["sort"])
        val_ds_gta = CacheGTAClassification(config["data_dir_gta"], split='val', cache_transform=transform, sort=config["sort"])
    else:
        train_ds_city = CacheCityscapesClassification(config["data_dir_cityscapes"], split='train', transform=transform, sort=config["sort"])
        val_ds_city = CacheCityscapesClassification(config["data_dir_cityscapes"], split='val', transform=transform, sort=config["sort"])   
        train_ds_gta = CacheGTAClassification(config["data_dir_gta"], split='train', transform=transform, sort=config["sort"])
        val_ds_gta = CacheGTAClassification(config["data_dir_gta"], split='val', transform=transform, sort=config["sort"])
    
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
        plt.title(f"Cityscapes (label {train_ds_city[i][1]})")
        plt.tight_layout()
        writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
        if i > 10:
            break
    for i in range(len(train_ds_gta)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(train_ds_gta[i][0]))
        plt.title(f"GTA5 (label {train_ds_gta[i][1]})")
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


    if config["clustering"] == "prior":
        # Perfect split between Cityscapes and GTA for each client
        # Final struct of the datasets and loaders dict: {cluster: {client: ds or loader}}
        train_ds_dict = {} 
        train_loader_dict = {}
        for c in clients:
            if "Cityscapes" not in local_samples[c]:
                train_ds_dict[c] = torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"])
            elif "GTA" not in local_samples[c]:
                train_ds_dict[c] = torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"])
            else:
                train_ds_dict[c] = ConcatDataset([
                    torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"]),
                    torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"])
                ])
            train_loader_dict[c] = torch.utils.data.DataLoader(train_ds_dict[c], batch_size=batch_size, shuffle=True, num_workers=4)
            
        # Simply one validation dataset and loader for each cluster, no further partitioning
        val_ds = ConcatDataset([val_ds_city, val_ds_gta])
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        
        example_client = clients[0]
        
        class_names = ["Cityscapes", "GTA"]
        
    elif config["clustering"] == "computed":
        # Build clustered local train datasets using clustering results 
        # Final structure : train_ds_dict = {cluster: {client: Concat(city, gta)}}
        train_ds_dict = {}
        train_loader_dict = {}
        
        with open(os.path.join(config["clustering_dir"], config["clustered_numbers"]), 'r') as f:
            clustered_numbers = json.load(f)
    
        # Merge institutional split and clustering results before creating datasets
        cluster_institution_numbers = merge_inst_and_cluster_numbers(local_samples, clustered_numbers)
        
        # Change the labels of the classification datasets to match the clustering results
        for cluster, cluster_datasets in cluster_institution_numbers.items():
                        
            for client, client_cluster_datasets in cluster_datasets.items():
                
                if "Cityscapes" in client_cluster_datasets:
                    train_ds_city.change_labels(client_cluster_datasets["Cityscapes"], int(cluster)) 
                if "GTA" in client_cluster_datasets:
                    train_ds_gta.change_labels(client_cluster_datasets["GTA"], int(cluster))     
         
        # Build the local dataset of each client
        for c in clients:
            if "Cityscapes" not in local_samples[c]:
                train_ds_dict[c] = torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"])
            elif "GTA" not in local_samples[c]:
                train_ds_dict[c] = torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"])
            else:
                train_ds_dict[c] = ConcatDataset([
                    torch.utils.data.Subset(train_ds_gta, local_samples[c]["GTA"]),
                    torch.utils.data.Subset(train_ds_city, local_samples[c]["Cityscapes"])
                ])
            train_loader_dict[c] = torch.utils.data.DataLoader(train_ds_dict[c], batch_size=batch_size, shuffle=True, num_workers=4)
            
        pprint(train_ds_dict)
        
        # Build validation set(s)
        if config["validation_mode"] == "oracle":
            
            with open(os.path.join(config["applied_clustering_dir"], config["clustered_numbers_val"]), 'r') as f:
                cluster_institution_numbers_val = json.load(f)
 
            for cluster, cluster_datasets in cluster_institution_numbers_val.items():
                
                if "Cityscapes" in cluster_datasets:
                    val_ds_city.change_labels(cluster_datasets["Cityscapes"], int(cluster))
                if "GTA" in cluster_datasets:
                    val_ds_gta.change_labels(cluster_datasets["GTA"], int(cluster))

            # Simply one validation dataset and loader for each cluster, no further partitioning
            val_ds = ConcatDataset([val_ds_city, val_ds_gta])
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

            example_client = clients[0]
    
            """
            print(val_ds_city[329][1], val_ds_city[55][1])
            print(val_ds_gta[9][1], val_ds_gta[203][1])
            sys.exit(0)  
            """
            
        class_names = list(cluster_institution_numbers_val.keys())
    
    train_sizes = {client:len(train_ds_dict[client]) for client in train_ds_dict}
    print("Train sizes: ", train_sizes)
    
    total_train_size = np.sum([len(d) for d in train_ds_dict.values()])
    print("Total train size: ", total_train_size)
    print("Val size: ", len(val_ds))
    
    """
    for im, labels in train_ds_dict["0"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["0"]["0"]))
    
    for im, labels in train_ds_dict["1"]["0"]:
        print(im.min(), im.max(), im.mean(), im.std())
    print(len(train_ds_dict["1"]["0"]))
    """
    pprint(train_ds_dict)
    pprint(train_loader_dict)
    im, labels = next(iter(train_loader_dict[example_client]))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
        
    device = torch.device("cuda:0")
    
    global_model = CNN_GTA_Cityscapes_source_Classification(sources=2).to(device)
    local_models = {client:CNN_GTA_Cityscapes_source_Classification(sources=2).to(device) for client in clients}
        
    print(global_model)
        
    local_optimizers = {client:optim.SGD(local_models[client].parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"]) for client in clients}
    
    if config["scaffold"]:
        # Define control variates
        client_c = {client:[] for client in clients}
        delta_c = {client:[] for client in clients}
        for client in clients:
            for param in global_model.parameters():
                client_c[client].append(torch.zeros_like(param.data))
                delta_c[client].append(torch.zeros_like(param.data))
            
        global_c = [torch.zeros_like(param.data) for param in global_model.parameters()]
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(global_model, next(iter(train_loader_dict[example_client]))[0].to(device))
    print(next(iter(train_loader_dict[example_client]))[0].size())
    summary(global_model, next(iter(train_loader_dict[example_client]))[0][0].size())
    
    # Get losses, metrics and optimizers
    loss_function = nn.CrossEntropyLoss()
    best_metric = 0.0
  
    for comm in range(config["max_epochs"]):  # loop over the dataset multiple times

        for client in train_ds_dict:
            
            for local_param, global_param in zip(local_models[client].parameters(), global_model.parameters()):
                local_param.data = global_param.data.clone()
                        
            epoch_loss = 0.0
            
            local_models[client].train()
            
            writer.add_scalar("Learning rate/Epoch", local_optimizers[client].param_groups[0]['lr'], comm+1)
            
            for idx, batch_data in enumerate(tqdm(train_loader_dict[client])):
    
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device).long()
                
                # zero the parameter gradients
                local_optimizers[client].zero_grad()
        
                # forward + backward + optimize
                output = local_models[client](inputs)
                    
                loss = loss_function(output, labels)
            
                loss.backward()
                    
                if config["scaffold"]:
                    # Alter local gradient with control variates
                    for param, global_control, local_control in zip(local_models[client].parameters(), global_c, client_c[client]):
                        param.grad.data += global_control - local_control
                        
                local_optimizers[client].step()
        
                # print statistics
                epoch_loss += loss.item()*len(inputs)
        
            if config["scaffold"]:
                for id_group, (client_c_group, global_c_group, local_model_group, global_model_group) in enumerate(zip(client_c[client], global_c, local_models[client].parameters(), global_model.parameters())):
                    delta_c[client][id_group] = -global_c_group + 1.0/len(train_loader_dict[client])/config["learning_rate"] * (global_model_group - local_model_group)
                    client_c[client][id_group] += delta_c[client][id_group]
                
            epoch_loss /= len(train_ds_dict[client])
        
            # Add loss value to tensorboard, and print it
            writer.add_scalar(f"Loss/train/Client {client}", epoch_loss, comm+1)
            print(f"\nClient {client}, comm {comm + 1}, average loss: {epoch_loss:.4f}")

        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        for client in train_ds_dict:  
            for global_param, client_param in zip(global_model.parameters(), local_models[client].parameters()):
                global_param.data += client_param.data.clone() * len(train_ds_dict[client])/total_train_size  
        
        if config["scaffold"]:
            # Then update global control variates using local control variates. For now, no weighting of control variates, maybe a mistake
            for client in clients:
                for id_group, delta_control_group in enumerate(delta_c[client]):
                    global_c[id_group] += delta_control_group * len(train_ds_dict[client])/total_train_size

        # Validation step
        if (comm) % config["validation_interval"] == 0:
            
            global_model.eval()
            
            with torch.no_grad():
                
                # Main loop on validation set
                for (idx, batch_data) in enumerate(tqdm(val_loader)):
                    
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    test_output = global_model(inputs)
                    _, pred_y = torch.max(test_output, 1, keepdim=True)
                    
                    if idx == 0:
                        total_truth = labels.detach().cpu().numpy()
                        total_pred = pred_y.detach().cpu().numpy()
                    else:
                        total_truth = np.concatenate([total_truth, labels.detach().cpu().numpy()], axis=0)
                        total_pred = np.concatenate([total_pred, pred_y.detach().cpu().numpy()], axis=0)
                    
                
                print(total_truth.shape, total_pred.shape)
                report = classification_report(total_truth, total_pred, target_names=class_names)
                
                report_dict = classification_report(total_truth, total_pred, target_names=class_names, output_dict=True)
                
                for cl in class_names:
                    writer.add_scalar(f"Validation/F1_score_class_{cl}", report_dict[cl]["f1-score"], comm + 1)
                writer.add_scalar("Validation/Weighted_average_F1_score", report_dict["weighted avg"]["f1-score"], comm + 1)
                
                print(report)
                
                if best_metric < report_dict["weighted avg"]["f1-score"]:
                    best_metric = report_dict["weighted avg"]["f1-score"]
                    torch.save(global_model.state_dict(), os.path.join(log_dir, "model.pth"))
                    print("\nSaved new best model")
                    
                    with open(os.path.join(log_dir, 'val_perf.json'), 'w') as f:
                        json.dump(report_dict, f, indent=4)  
    
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
