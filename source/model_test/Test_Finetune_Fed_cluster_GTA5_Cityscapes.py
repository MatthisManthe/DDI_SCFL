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
    
    if config["oracle_eval"]:
        with open(os.path.join(config["clustering_dir"], config["clustered_images"]), 'r') as f:
            cluster_part_image_train = json.load(f)
            
        clusters = list(cluster_part_image_train.keys())
    else:
        clusters = config["clusters"]
        
        
    if config["validation_mode"] == "on_IID":
        
        if config["model"] == "EfficientVIT":
            # Build the model and its parallel counterpart
            if config["labels"] == "trainId":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for cluster in clusters}
                
            elif config["labels"] == "categories":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for cluster in clusters}
    
        for cluster in clusters:
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(config['clustered_models_dir'], config["model_file"].replace("#", cluster))))
            
        print(cluster_models[list(cluster_models.keys())[0]])
        
        with open(os.path.join(config["test_IID_split_path"]), 'r') as f:
            test_iid_samples = json.load(f)
        
        clients = list(test_iid_samples.keys())
        
        local_datasets = {cluster:[] for cluster in clusters} 
        
        for idc, c in enumerate(clients):
            
            clust = clusters[0] if idc < 5 else clusters[1]
            local_datasets[clust].append(ConcatDataset([
                torch.utils.data.Subset(val_ds_city, test_iid_samples[c]["Cityscapes"]),
                torch.utils.data.Subset(val_ds_gta, test_iid_samples[c]["GTA"])
            ]))
        
        local_datasets[clusters[0]] = ConcatDataset(local_datasets[clusters[0]])
        local_datasets[clusters[1]] = ConcatDataset(local_datasets[clusters[1]])
        
        val_local_loaders = {c:torch.utils.data.DataLoader(local_datasets[c], batch_size=batch_size, shuffle=True, num_workers=0) for c in clusters}

        # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
        writer.add_graph(cluster_models["Cityscapes"], next(iter(val_loader))[0].to(device))
        print(next(iter(val_loader))[0].size())
        #summary(cluster_models["0"], next(iter(train_loader_dict["0"]["0"]))[0][0].size())
    
        # Get losses, metrics and optimizers
        metric_function = CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)
        for cluster in clusters:
            cluster_models[cluster].eval()
            
            with torch.no_grad():

                for (idx, batch_data) in enumerate(tqdm(val_local_loaders[cluster])):
                    
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    outputs = cluster_models[cluster](inputs)
                    if outputs.shape[-2:] != labels.shape[-2:]:
                        outputs = resize(outputs, size=labels.shape[-2:])
                    _, prediction = torch.max(outputs.data, 1, keepdim=True)
                    
                    metric_function(prediction, labels)
                 
        report = metric_function.aggregate()
        metric_function.reset()
        
        print(report)

        with open(os.path.join(log_dir, f'final_{config["split"]}_prior_inference_results_on_IID_Split.json'), 'w') as fp:
            json.dump(report, fp, indent=4)
            
        with open(os.path.join(config["clustered_models_dir"], f'final_{config["split"]}_prior_inference_results_on_IID_Split.json'), 'w') as fp:
            json.dump(report, fp, indent=4)
            
            
    elif config["validation_mode"] == "mcdropout":
        
        if config["model"] == "EfficientVIT":
            # Build the model and its parallel counterpart
            if config["labels"] == "trainId":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for cluster in clusters}
                
            elif config["labels"] == "categories":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"], dropout=config["dropout"]).to(device) for cluster in clusters}
    
        for cluster in clusters:
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(config['clustered_models_dir'], config["model_file"].replace("#", cluster))))
            
        print(cluster_models[list(cluster_models.keys())[0]])
        
        # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
        writer.add_graph(cluster_models["Cityscapes"], next(iter(val_loader))[0].to(device))
        print(next(iter(val_loader))[0].size())
        #summary(cluster_models["0"], next(iter(train_loader_dict["0"]["0"]))[0][0].size())
        
        # Get losses, metrics and optimizers
        metric_functions_variance = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
        metric_functions_uncertainty = [CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)]
        metrics = {}
    
        for cluster in clusters:
            cluster_models[cluster].train()
           
        os.mkdir(os.path.join(log_dir, config["result_folder_name"]))
        selected_cluster = []
        selected_cluster_variance = []
        
        data_origin = np.array([0]*len(val_ds_city)+[1]*len(val_ds_gta))
        
        with torch.no_grad():
            
            for (idx, batch_data) in enumerate(tqdm(val_loader)):
                
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                cluster_probs = []
                cluster_outputs = []
                cluster_uncertainty_maps = []
                cluster_full_uncertainty = []
                cluster_variance_maps = []
                cluster_full_variance = []
                
                for cluster in clusters:
    
                    outputs = []
                    
                    for mc_sample in range(config["nb_mc_samples"]):
                        outputs.append(cluster_models[cluster](inputs))
                        
                        if config["model"] == "EfficientVIT" and outputs[-1].shape[-2:] != labels.shape[-2:]:
                            outputs[-1] = resize(outputs[-1], size=labels.shape[-2:])
                    
                    #print(len(outputs), outputs[0].shape)
                    
                    probs = torch.stack([F.softmax(outputs[i], dim=1).detach() for i in range(config["nb_mc_samples"])], dim=0)
                    average_output = probs.mean(0)
                    uncertainty_map = scipy.stats.entropy(average_output.cpu().numpy(), axis=1, base=outputs[0].shape[1])
                    print("\n\n\nProb shape: ", probs.shape)
                    variance_map = torch.std(probs, dim=0).mean(dim=1)
    
                    print("\n\”Variance shape: ", variance_map.shape)
                    probs = torch.transpose(probs, 0, 1)
                    full_uncertainty = np.mean(uncertainty_map, axis=(-1, -2))
                    full_variance = np.mean(variance_map.detach().cpu().numpy(), axis=(-1, -2))
                    print(full_variance.shape, probs.shape)
                    #print("\n", probs.shape, average_output.shape, uncertainty_map.shape, full_uncertainty)
                    
                    cluster_probs.append(probs)
                    cluster_outputs.append(average_output)
                    cluster_uncertainty_maps.append(uncertainty_map)
                    cluster_full_uncertainty.append(full_uncertainty)
                    cluster_variance_maps.append(variance_map)
                    cluster_full_variance.append(full_variance)
                 
                #print("\Memory usage after cluster inferences: ", psutil.virtual_memory().percent)
                # Wrng weight computation, connard
                cluster_weights = torch.tensor(np.array(cluster_full_uncertainty)/np.sum(cluster_full_uncertainty, axis=0, keepdims=True), dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)            
                cluster_outputs = torch.stack(cluster_outputs, dim=0).to(device)
                #print("\n", cluster_outputs.shape, cluster_weights.shape)
                
                _, final_prediction = torch.max(torch.sum(torch.tensor(cluster_outputs) * cluster_weights, dim=0), 1, keepdim=True)
                _, cluster_outputs = torch.max(cluster_outputs, 2, keepdim=True)
                #print("\n", final_prediction.shape, labels.shape, cluster_outputs.shape)
                
                cluster_uncertainty_maps = torch.stack([torch.tensor(c) for c in cluster_uncertainty_maps], dim=0)
                cluster_variance_maps = torch.stack([torch.tensor(c) for c in cluster_variance_maps], dim=0)
                print(cluster_variance_maps.shape)
                #print(cluster_uncertainty_maps.shape)
                
                cluster_full_uncertainty = np.array(cluster_full_uncertainty)
                cluster_full_variance = np.array(cluster_full_variance)
                print("Cluster full uncertainty: ", cluster_full_uncertainty)
                print("Cluster full variance: ", cluster_full_variance)
                batch_selected_cluster = np.argmin(cluster_full_uncertainty, axis=0)
                batch_selected_cluster_variance = np.argmin(cluster_full_variance, axis=0)
                selected_cluster.extend(list(batch_selected_cluster))
                selected_cluster_variance.extend(list(batch_selected_cluster_variance))
                
                uncertainty_based_final_output = torch.stack([cluster_outputs[batch_selected_cluster[b], b] for b in range(len(batch_selected_cluster))])
                variance_based_final_output = torch.stack([cluster_outputs[batch_selected_cluster_variance[b], b] for b in range(len(batch_selected_cluster_variance))])
                
                print("RESULTS SIZES TO WATCH FOR : ", cluster_outputs.shape, uncertainty_based_final_output.shape, variance_based_final_output.shape, labels.shape)
                metric_functions_uncertainty[0](uncertainty_based_final_output, labels)
                metric_functions_variance[0](variance_based_final_output, labels)
                
                
                #plt.style.use("plot_styles/subplots_style.mplstyle")
                plt.style.use("/gpfswork/rech/geh/uzq69ur/advanced_sample_level_cffl/source/plot_styles/subplots_style.mplstyle")
                
                for i in range(len(inputs)):
                    # Only save the image if the cluster selection was wrong for one of the methods.
                    if batch_selected_cluster[i] != data_origin[idx*config["batch_size"]+i] or batch_selected_cluster_variance[i] != data_origin[idx*config["batch_size"]+i]:
                        nb_plotted_samples = int(min(4,config["nb_mc_samples"]))
                        fig, axs = plt.subplots(4+nb_plotted_samples, 2, figsize=(14,14+3*nb_plotted_samples))
                        #fig, axs = plt.subplots(4+nb_plotted_samples, 2)
                        axs = axs.ravel()
                        #print(inputs[i].shape, labels[i].shape, final_prediction[i].shape)
                        
                        fig.subplots_adjust(right=0.98)
        
                        axs[0].imshow(prepare_plot_im(inputs[i]))
                        axs[0].imshow(prepare_plot_label(labels[i][0], labels=config["labels"]), alpha=0.7)
                        axs[0].title.set_text('Ground truth')
                        
                        axs[1].imshow(prepare_plot_im(inputs[i]))
                        axs[1].imshow(prepare_plot_label(final_prediction[i][0], labels=config["labels"]), alpha=0.7)
                        averaged_prediction_score = sample_mIoU_Cityscapes(final_prediction[i], labels[i], label_values, config["labels"])
                        axs[1].title.set_text(f'Uncertainty averaged prediction (mIoU: {averaged_prediction_score:.4})')
                        
                        axs[2].imshow(prepare_plot_im(inputs[i]))
                        axs[2].imshow(prepare_plot_label(cluster_outputs[0][i][0], labels=config["labels"]), alpha=0.7)
                        cluster_0_prediction_score = sample_mIoU_Cityscapes(cluster_outputs[0][i], labels[i], label_values, config["labels"])
                        axs[2].title.set_text(f'Cluster 0 prediction (mIoU: {cluster_0_prediction_score:.4})')
                        
                        axs[3].imshow(prepare_plot_im(inputs[i]))
                        axs[3].imshow(prepare_plot_label(cluster_outputs[1][i][0], labels=config["labels"]), alpha=0.7)
                        cluster_1_prediction_score = sample_mIoU_Cityscapes(cluster_outputs[1][i], labels[i], label_values, config["labels"])
                        axs[3].title.set_text(f'Cluster 1 prediction (mIoU: {cluster_1_prediction_score:.4})')
                        
                        axs[4].imshow(prepare_plot_im(inputs[i]))
                        axs[4].imshow(cluster_uncertainty_maps[0][i].detach().cpu().numpy(), alpha=0.5, cmap="hot", vmin=0, vmax=1)
                        axs[4].title.set_text(f'Cluster 0 uncertainty map (Averaged uncertainty: {cluster_full_uncertainty[0][i]:.4})')
                        
                        axs[5].imshow(prepare_plot_im(inputs[i]))
                        im = axs[5].imshow(cluster_uncertainty_maps[1][i].detach().cpu().numpy(), alpha=0.5, cmap="hot", vmin=0, vmax=1)
                        axs[5].title.set_text(f'Cluster 1 uncertainty map (Averaged uncertainty: {cluster_full_uncertainty[1][i]:.4})')
                        
                        bbox = axs[5].get_position()
                        cbar_ax = fig.add_axes([0.98, bbox.y0, 0.01, bbox.y1-bbox.y0])
                        fig.colorbar(im, cax=cbar_ax)
                        
                        axs[6].imshow(prepare_plot_im(inputs[i]))
                        axs[6].imshow(cluster_variance_maps[0][i].detach().cpu().numpy(), alpha=0.5, cmap="hot", vmin=0, vmax=0.3)
                        axs[6].title.set_text(f'Cluster 0 std map (Averaged std: {cluster_full_variance[0][i]:.4})')
                        
                        axs[7].imshow(prepare_plot_im(inputs[i]))
                        im = axs[7].imshow(cluster_variance_maps[1][i].detach().cpu().numpy(), alpha=0.5, cmap="hot", vmin=0, vmax=0.3)
                        axs[7].title.set_text(f'Cluster 1 std map (Averaged std: {cluster_full_variance[1][i]:.4})')
                        
                        bbox = axs[7].get_position()
                        cbar_ax = fig.add_axes([0.98, bbox.y0, 0.01, bbox.y1-bbox.y0])
                        fig.colorbar(im, cax=cbar_ax)
                        
                        for j in range(nb_plotted_samples):
                            axs[8+2*j].imshow(prepare_plot_im(inputs[i]))
                            _, output = torch.max(cluster_probs[0][i][j], 0, keepdim=True)
                            axs[8+2*j].imshow(prepare_plot_label(output[0], labels=config["labels"]), alpha=0.7)
                            cluster_0_prediction_score = sample_mIoU_Cityscapes(output, labels[i], label_values, config["labels"])
                            axs[8+2*j].title.set_text(f'Cluster 0 prediction {j} (mIoU: {cluster_0_prediction_score:.4})')
                            
                            axs[9+2*j].imshow(prepare_plot_im(inputs[i]))
                            _, output = torch.max(cluster_probs[1][i][j], 0, keepdim=True)
                            axs[9+2*j].imshow(prepare_plot_label(output[0], labels=config["labels"]), alpha=0.7)
                            cluster_1_prediction_score = sample_mIoU_Cityscapes(output, labels[i], label_values, config["labels"])
                            axs[9+2*j].title.set_text(f'Cluster 1 prediction {j} (mIoU: {cluster_1_prediction_score:.4})')
                        
                        sample_path = val_ds_city.image_paths[idx*config["batch_size"]+i] if idx*config["batch_size"]+i < min_dataset_size_val else val_ds_gta.image_paths[idx*config["batch_size"]+i-min_dataset_size_val]
                        fig.suptitle(f"{sample_path}")
                        
                        plt.savefig(os.path.join(log_dir, config["result_folder_name"], os.path.basename(sample_path)), bbox_inches='tight')
                        plt.close("all")
    
            found_origin = np.array(selected_cluster)
            found_origin_variance = np.array(selected_cluster_variance)
            
            rand_ind = rand_score(data_origin, found_origin)
            print("Rand index: ", rand_ind)
            
            pd.crosstab(data_origin, found_origin).to_csv(os.path.join(log_dir, "Max_uncertainty_cluster_selection_confusion_matrix.csv"))
            pd.DataFrame({"rand_index":rand_ind}, index=[0]).to_csv(os.path.join(log_dir, "Max_uncertainty_cluster_selection_rand_index.csv"))
            
            rand_ind = rand_score(data_origin, found_origin_variance)
            print("Rand index: ", rand_ind)
            
            pd.crosstab(data_origin, found_origin_variance).to_csv(os.path.join(log_dir, "Max_variance_cluster_selection_confusion_matrix.csv"))
            pd.DataFrame({"rand_index":rand_ind}, index=[0]).to_csv(os.path.join(log_dir, "Max_variance_cluster_selection_rand_index.csv"))
    
            report_uncertainty = metric_functions_uncertainty[0].aggregate()
            report_variance = metric_functions_variance[0].aggregate()
            
            print("Uncertainty based: ", report_uncertainty)
            print("Variance based: ", report_variance)
            
            with open(os.path.join(log_dir, f'final_{config["split"]}_results_uncertainty_based.json'), 'w') as fp:
                json.dump(report_uncertainty, fp, indent=4)
                
            with open(os.path.join(log_dir, f'final_{config["split"]}_results_variance_based.json'), 'w') as fp:
                json.dump(report_variance, fp, indent=4)
                
            with open(os.path.join(config["clustered_models_dir"], f'final_{config["split"]}_results_uncertainty_based.json'), 'w') as fp:
                json.dump(report_uncertainty, fp, indent=4)
                
            with open(os.path.join(config["clustered_models_dir"], f'final_{config["split"]}_results_variance_based.json'), 'w') as fp:
                json.dump(report_variance, fp, indent=4)
        
        
    elif config["validation_mode"] == "prior":
        
        if config["model"] == "EfficientVIT":
            # Build the model and its parallel counterpart
            if config["labels"] == "trainId":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for cluster in clusters}
                
            elif config["labels"] == "categories":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for cluster in clusters}
    
        for cluster in clusters:
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(config['clustered_models_dir'], config["model_file"].replace("#", cluster))))
            
        print(cluster_models[list(cluster_models.keys())[0]])
        
        # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
        writer.add_graph(cluster_models["Cityscapes"], next(iter(val_loader))[0].to(device))
        print(next(iter(val_loader))[0].size())
        #summary(cluster_models["0"], next(iter(train_loader_dict["0"]["0"]))[0][0].size())
    
        # Get losses, metrics and optimizers
        metric_function = CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)
        for cluster in clusters:
            cluster_models[cluster].eval()
            
        with torch.no_grad():
            
            for (idx, batch_data) in enumerate(tqdm(val_loader_city)):
                
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                outputs = cluster_models["Cityscapes"](inputs)
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = resize(outputs, size=labels.shape[-2:])
                _, prediction = torch.max(outputs.data, 1, keepdim=True)
                
                metric_function(prediction, labels)
                
            for (idx, batch_data) in enumerate(tqdm(val_loader_gta)):
                
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                outputs = cluster_models["GTA"](inputs)
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = resize(outputs, size=labels.shape[-2:])
                _, prediction = torch.max(outputs.data, 1, keepdim=True)
                
                metric_function(prediction, labels)
                 
        report = metric_function.aggregate()
        metric_function.reset()
        
        print(report)

        with open(os.path.join(log_dir, f'final_{config["split"]}_prior_inference_results.json'), 'w') as fp:
            json.dump(report, fp, indent=4)
            
        with open(os.path.join(config["clustered_models_dir"], f'final_{config["split"]}_prior_inference_results.json'), 'w') as fp:
            json.dump(report, fp, indent=4)
    
    
    elif config["validation_mode"] == "source_classifier":
        
        if config["model"] == "EfficientVIT":
            # Build the model and its parallel counterpart
            if config["labels"] == "trainId":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for cluster in clusters}
                
            elif config["labels"] == "categories":
                cluster_models = {cluster:seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device) for cluster in clusters}
    
        for cluster in clusters:
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(config['clustered_models_dir'], config["model_file"].replace("#", cluster))))
         
        source_classifier = CNN_GTA_Cityscapes_source_Classification(sources=2).to(device)
        source_classifier.load_state_dict(torch.load(config["source_classifier_path"]))
         
        metric_function = CumulativeIoUCityscapes(num_classes=nb_classes, label_type=config["labels"], ignore_background=ignore_background)
        for cluster in clusters:
            cluster_models[cluster].eval()
          
        with torch.no_grad():
            
            for (idx, batch_data) in enumerate(tqdm(val_loader)):
                
                inp, label = batch_data[0].to(device), batch_data[1].to(device)
                
                source_classif_output = source_classifier(inp)
                _, source_prediction = torch.max(source_classif_output, 1)
                
                output = cluster_models[str(source_prediction.item())](inp)
                if output.shape[-2:] != labels.shape[-2:]:
                    output = resize(output, size=labels.shape[-2:])
                _, prediction = torch.max(output.data, 1, keepdim=True)
                
                metric_function(prediction, label)
                
        report = metric_function.aggregate()
        metric_function.reset()
        
        print(report)

        with open(os.path.join(log_dir, f'final_{config["split"]}_prior_inference_results.json'), 'w') as fp:
            json.dump(report, fp, indent=4)
            
        with open(os.path.join(config["clustered_models_dir"], f'final_{config["split"]}_prior_inference_results.json'), 'w') as fp:
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
