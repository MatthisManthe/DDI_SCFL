import torch
import sys
import os
import json
import time
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
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from torch.nn.functional import normalize
import pickle 
from pprintpp import pprint
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.cluster import rand_score
from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd
import pacmap
from sklearn.manifold import TSNE
from monai.transforms import NormalizeIntensity, ToTensor
from monai.networks.nets import DynUNet

import matplotlib.lines as mlines
import matplotlib
import monai 

from efficientvit import seg_model_zoo
from efficientvit.models.utils import resize
from plot_utils import imshow
from torchvision.utils import draw_segmentation_masks
from data.cityscapes_labels import CsLabels
from data.datasets import CacheCityscapes, CacheGTA
from data.transforms import prepare_plot_im, prepare_plot_label, \
    generate_transform_cityscapes_im, generate_transform_cityscapes_label, generate_transform_GTA5_label
from models.models import modified_get_conv_layer
from metrics.metrics import CumulativeClassificationMetrics, CumulativeIoUCityscapes
from data.partitioning_utils import assert_GTA_Cityscapes_partitioning

gaussian_weights = None
similarity_mat = None

test_data_interactive = None
id_color = 0
colors = ['r', 'g', 'b', 'y']
shade_color = ["mistyrose", "lightgreen", "lightblue", "lightyellow"]

def close_factors(number):
    ''' 
    find the closest pair of factors for a given number
    '''
    factor1 = 0
    factor2 = number
    while factor1 +1 <= factor2:
        factor1 += 1
        if number % factor1 == 0:
            factor2 = number // factor1
        
    return factor1, factor2

def almost_factors(number):
    '''
    find a pair of factors that are close enough for a number that is close enough
    '''
    while True:
        factor1, factor2 = close_factors(number)
        if 1/2 * factor1 <= factor2: # the fraction in this line can be adjusted to change the threshold aspect ratio
            break
        number += 1
    return factor1, factor2


def onpick_City(event):
    
    global test_data_interactive, id_color, colors

    n = len(event.ind)
    if not n:
        return
    print(event.ind)
    
    pos = event.artist._offsets[event.ind]
    pick_x = event.mouseevent.xdata
    pick_y = event.mouseevent.ydata
    
    s_x = max([abs(x - pick_x) for [x,_] in pos])
    s_y = max([abs(y - pick_y) for [_,y] in pos])
    
    #circle1 = plt.Circle((pick_x, pick_y), radius=s+0.003, color=colors[id_color], fill=False)
    ellipse = matplotlib.patches.Ellipse((pick_x, pick_y), height=2*s_y, width=2*s_x, fill=False)
    event.mouseevent.inaxes.add_patch(ellipse)
    plt.draw()

    row, col = almost_factors(n)
    #fig, axs = plt.subplots(row, col, squeeze=False, facecolor=shade_color[id_color], figsize=(15,15))
    fig, axs = plt.subplots(row, col, squeeze=False, figsize=(15,15))
    for dataind, ax in zip(event.ind, axs.flat):
        ax.imshow(prepare_plot_im(test_data_interactive[dataind][0]))
        ax.imshow(prepare_plot_label(test_data_interactive[dataind][1][0]), alpha=0.2)
    fig.tight_layout()
    fig.show()

    return True


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


# ---------------- Gradient computation and distances -----------------
def flatten(params):
    return torch.cat([param.flatten() for param in params])


def similarity_gmm(x, y, classes, config, per_class_similarity="bhattacharya"):
    
    sim = 0
    num_classes = len(classes)
    num_cluster = config["nb_clusters"]
    num_overlap = 0
    
    for c in range(num_classes):
        if x[c*num_cluster:(c+1)*num_cluster].sum() > 0 and y[c*num_cluster:(c+1)*num_cluster].sum() > 0:
            sim += np.dot(x[c*num_cluster:(c+1)*num_cluster], y[c*num_cluster:(c+1)*num_cluster])
            num_overlap += 1
            
    return sim/num_overlap if num_overlap != 0 else 0.


# ------------------ Dimensionality reduction ----------------------
def std_parameter_selection(data, model, config, log_dir, name):
    
    print("Computing parameter variances and pruning")
    
    std = np.std(data, axis=0)
    df = pd.DataFrame(std)
    df.to_csv(os.path.join(log_dir, f"{name}_std_per_parameter.csv"))
    
    selected_parameters = np.argsort(std)[-config["dimension"]:]
    data = data[:,selected_parameters]
    
    plot_selected_parameters(std, selected_parameters, model, log_dir, name)
    
    print("New data shape: ", data.shape)
    
    return data, selected_parameters


def norm_std_parameter_selection(data, model, config, log_dir, name):
    
    print("Computing parameter variances and pruning")
        
    std = np.std(data, axis=0)
    param_groups_nb_param = [param.numel() for _, param in model.named_parameters()]
    param_groups_param_range = [0] + list(np.cumsum(param_groups_nb_param))
    # Counteract kaiming init ?
    for i in range(len(param_groups_param_range) - 1):
        std[param_groups_param_range[i]:param_groups_param_range[i+1]] *= np.sqrt(param_groups_nb_param[i]) 
    
    df = pd.DataFrame(std)
    df.to_csv(os.path.join(log_dir, f"{name}_std_per_parameter.csv"))
    
    selected_parameters = np.argsort(std)[-config["dimension"]:]
    data = data[:,selected_parameters]
    
    plot_selected_parameters(std, selected_parameters, model, log_dir, name)
    
    print("New data shape: ", data.shape)
    
    return data, selected_parameters
    
    
def random_parameter_selection(data, model, config, log_dir, name):
    
    print(f"Selecting {config['dimension']} parameters from the model at random.")
    
    selected_parameters = np.random.choice(np.arange(data.shape[1]), size=config["dimension"], replace=False)
    data = data[:,selected_parameters]
    
    plot_selected_parameters(None, selected_parameters, model, log_dir, name)
    
    print("New data shape: ", data.shape)
    
    return data, selected_parameters
    
    
def pca_reduction(data, config):
    
    print("Performing PCA before clustering")
    
    pca = PCA(n_components=config["dimension"])
    data = pca.fit_transform(data)
    
    print("New data shape: ", data.shape)
    
    return data
    
    
def truncatedsvd_reduction(data, config):
    
    print("Performing PCA before clustering")
    
    svd = TruncatedSVD(n_components=config["dimension"])
    data = svd.fit_transform(data)
    
    print("New data shape: ", data.shape)
    
    return data
    

# -------------------------- Plot utils ------------------------------
def plot_selected_parameters(std, selected_parameters, model, log_dir, name):
    
    param_groups = [name for name, _ in model.named_parameters()]
    param_groups_nb_param = [param.numel() for _, param in model.named_parameters()]
    param_groups_param_range = [0] + list(np.cumsum(param_groups_nb_param))
    param_group_norm = [torch.norm(param.data).item() for param in model.parameters()]
    
    nb_selected_param_per_group = [len([idx for idx in selected_parameters if (idx >= param_groups_param_range[i] and idx < param_groups_param_range[i+1])]) for i in range(len(param_groups_param_range) - 1)]
    percentage_selected_param_per_group = [(nb_selected / nb) * 100. for nb, nb_selected in zip(param_groups_nb_param, nb_selected_param_per_group)]
    
    plt.figure(figsize=(80,12))
    plt.bar(param_groups, nb_selected_param_per_group)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{name}_nb_selected_parameters_per_group.svg"))
    
    plt.figure(figsize=(80,12))
    plt.bar(param_groups, percentage_selected_param_per_group)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{name}_percentage_selected_parameters_per_group.svg"))
    
    plt.figure(figsize=(80,12))
    plt.bar(param_groups, param_group_norm)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{name}_group_norm.svg"))
    
    plt.figure(figsize=(80,12))
    plt.bar(param_groups, param_groups_nb_param)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{name}_nb_param_per_group.svg"))
    
    plt.figure(figsize=(80,12))
    plt.bar(param_groups, param_groups_nb_param)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{name}_nb_param_per_group.svg"))
    
    if std is not None:
        
        param_group_std = [np.mean(std[param_groups_param_range[i]:param_groups_param_range[i+1]]) for i in range(len(param_groups_param_range) - 1)]
        
        plt.figure(figsize=(80,12))
        plt.bar(param_groups, param_group_std)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"{name}_average_std_per_group.svg"))
    
    
    

def origin_vs_cluster_pacmap(data, true_label, predicted_labels, config, log_dir, name):
    
    # Plot PaCMAP of clustered gradients
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, verbose=True, random_state=config["seed"]) 
    X_transformed = embedding.fit_transform(data, init="pca")
    
    markers = ["o", "1", "v", "<", ">", "8", "*", "+", "D"]
    colors = ["b", "r", "yellow", "green", "orange", "gray", "brown", "purple"]
    
    plot_label = ["cluster "+str(label)+" origin "+str(origin) for origin, label in zip(true_label, predicted_labels)]
    color_samples = [colors[o] for o in true_label]
    mark_samples = [markers[l] for l in predicted_labels]
    
    # Create a scatter plot
    fig = plt.figure(figsize=(9, 9))
    mscatter(X_transformed[:,0], X_transformed[:,1], s=7, label=plot_label, c=color_samples, m=mark_samples, picker=True, pickradius=config["pick_radius"])
    
    handles = []
    for id_o, origin in enumerate(np.unique(true_label)):
        for id_c, cluster in enumerate(np.unique(predicted_labels)):
            handles.append(mlines.Line2D([], [], color=colors[id_o], 
                                         marker=markers[id_c], 
                                         markersize=5, label="cluster "+str(cluster)+" origin "+str(origin),
                                         linestyle='None'))
    #plt.legend(handles=handles)
    
    plt.title(f"PaCMAP {name}")
    plt.axis('tight')
    
    plt.savefig(os.path.join(log_dir, f"{name}_Pacmap.svg"))
    
    
def plot_cluster_class_distribution(predicted_labels, classes, class_distribution, log_dir):
    
    # Plot class distribution per cluster
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        xlabel='Class',
        ylabel='Nb pixels',
    )
    
    cluster_colors = ["black", "red", "blue", "green", "yellow", "orange", "gray", "brown"]
    class_space = 3
    inter_space = 2./config["nb_clusters"]
    for idcluster in range(config["nb_clusters"]):

        for idclass, c in enumerate(classes):
            
            cluster_samples_id = np.argwhere(predicted_labels == idcluster).flatten()
            """
            print(len(cluster_samples_id))
            print(cluster_samples_id.shape)
            print(class_distribution.shape)
            print((class_distribution[cluster_samples_id]).shape)
            print(((class_distribution[cluster_samples_id])[:,idclass]).shape)
            """
            bp = ax1.boxplot(
                class_distribution[cluster_samples_id][:,idclass], sym='+',
                positions=[class_space*idclass-inter_space*(config["nb_clusters"] - 1)/2.+inter_space*idcluster]
            )
            plt.setp(bp['boxes'], color=cluster_colors[idcluster])
            plt.setp(bp['whiskers'], color=cluster_colors[idcluster])
            plt.setp(bp['fliers'], color=cluster_colors[idcluster], marker='+')
    
    plt.xticks(np.arange(len(classes))*class_space, classes)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "Class_distribution_plain_gradient_clustering.svg"))



def main(args, config):
    """Main function"""
    
    global test_data_interactive
    
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
    
    transform_city = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_cityscapes_label(size=config["size"], labels=config["labels"]))
    transform_gta = (generate_transform_cityscapes_im(size=config["size"]), generate_transform_GTA5_label(size=config["size"], labels=config["labels"]))
    
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
    
    print("just before cache")
    if config["cache"]:
        ds_city.cache()
        ds_gta.cache()
    
    dataset_size = len(ds_city)
    
    loader_city = torch.utils.data.DataLoader(ds_city, batch_size=1, 
                                              shuffle=False, num_workers=config["num_workers"])
    loader_gta = torch.utils.data.DataLoader(ds_gta, batch_size=1, 
                                              shuffle=False, num_workers=config["num_workers"])
    
    print(ds_city[0][0].min(), ds_city[0][0].max(), ds_city[0][0].mean(), ds_city[0][0].std())
    print(ds_gta[0][0].min(), ds_gta[0][0].max(), ds_gta[0][0].mean(), ds_gta[0][0].std())
    
    print(ds_city, ds_gta)
    print(len(ds_city), len(ds_gta))
    print(ds_city[0][0].size(), ds_gta[0][0].size())
    fig, axs = plt.subplots(2,1, figsize=(15,15))
    axs[0].imshow(prepare_plot_im(ds_city[0][0]))
    axs[1].imshow(prepare_plot_im(ds_city[0][0]))
    axs[1].imshow(prepare_plot_label(ds_city[0][1][0], labels=config["labels"]), alpha=0.7)    
    plt.savefig(os.path.join(log_dir, "cityscapes_sample.svg"))
    plt.show()
    
    fig, axs = plt.subplots(2,1, figsize=(15,15))
    axs[0].imshow(prepare_plot_im(ds_gta[0][0]))
    axs[1].imshow(prepare_plot_im(ds_gta[0][0]))
    axs[1].imshow(prepare_plot_label(ds_gta[0][1][0], labels=config["labels"]), alpha=0.7)
    plt.savefig(os.path.join(log_dir, "gta_sample.svg"))
    plt.show()
    
    print(ds_city[0][0].min(), ds_city[0][0].max(), ds_city[0][0].mean(), ds_city[0][0].std())
    print(ds_gta[0][0].min(), ds_gta[0][0].max(), ds_gta[0][0].mean(), ds_gta[0][0].std())
    
    if config["labels"] == "trainId":
        classes = list(range(19))
        if config["model"] == "UNet":
            out_channels = 20
        elif config["model"] == "EfficientVIT":
            out_channels = 19
    elif config["labels"] == "categories":
        classes = list(range(7))
        out_channels = 7

    if config["model"] == "EfficientVIT":
        if config["labels"] == "trainId":
            model = seg_model_zoo.create_seg_model("b0", "cityscapes", pretrained=False, norm=config["norm"]).to(device)
        elif config["labels"] == "categories":
            model = seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device)
    
    if not config["rand_init"]:
        model.load_state_dict(torch.load(os.path.join(config["model_dir"], config["trained_model_file"])))
        
    data_origin = np.array([0]*dataset_size+[1]*dataset_size)
    print("Data_origin: ", data_origin)  
        
    if config["subset_classes"] != 0:
        classes = classes[:config["subset_classes"]]
        
    os.mkdir(os.path.join(log_dir, "per_class_clustering"))
    
    # Compute class-wise average gradients
    if config["labels"] == "trainId":
        gradient_list_per_class = {c:{"name": CsLabels.trainId2label[c].name, "gradients":np.array([]), "mapping":[]} for c in classes}
    elif config["labels"] == "categories":
        gradient_list_per_class = {c:{"name": CsLabels.category2label[c].category, "gradients":np.array([]), "mapping":[]} for c in classes}
      
    model.train()
    
    
    
    
    # Plain gradient clustering, see what it does (should not be perfect, affected by class distribution in images)
    if config["clustering"] == "CFL":
        
        with open(os.path.join(config["partitioning_file"]), 'r') as f:
            local_samples = json.load(f)
        
        clients = list(local_samples.keys())
        
        # Assure that we use a correct partitioning of the data
        assert_GTA_Cityscapes_partitioning(local_samples, clients)
        
        local_datasets = {} 
        for c in clients:
            if "Cityscapes" not in local_samples[c]:
                local_datasets[c] = torch.utils.data.Subset(ds_gta, local_samples[c]["GTA"])
            elif "GTA" not in  local_samples[c]:
                local_datasets[c] = torch.utils.data.Subset(ds_city, local_samples[c]["Cityscapes"])
            else:
                local_datasets[c] = ConcatDataset([
                    torch.utils.data.Subset(ds_city, local_samples[c]["Cityscapes"]),
                    torch.utils.data.Subset(ds_gta, local_samples[c]["GTA"])
                ])
            
        local_loaders = {c:torch.utils.data.DataLoader(local_datasets[c], batch_size=config["batch_size"], shuffle=True, num_workers=0) for c in clients}
        loss_function = nn.CrossEntropyLoss(ignore_index=255)
        learning_rate = config["learning_rate"]*(config["gamma"]**config["split_round"])
        
        local_model = seg_model_zoo.create_seg_model("b0_categories", "cityscapes", pretrained=False, norm=config["norm"]).to(device)
        local_optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        updates = {client:[torch.zeros_like(param) for param in model.parameters()] for client in clients}
        nb_clients = len(clients)
        
        for client, loader in local_loaders.items():
            
            for local_param, global_param in zip(local_model.parameters(), model.parameters()):
                local_param.data = global_param.data.clone()
                
            local_model.train()
            
            for i, (images, labels) in enumerate(tqdm(local_loaders[client])):
    
                inputs, labels = images.to(device), labels.long().to(device)
                
                outputs = local_model(inputs)
                if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = resize(outputs, size=labels.shape[-2:])
                loss = loss_function(outputs, labels.squeeze(1))
                
                # clear gradients for this training step   
                local_optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()        
                
                # apply gradients             
                local_optimizer.step()          
    
            for id_group, (old_param, new_param) in enumerate(zip(model.parameters(), local_model.parameters())):
                updates[client][id_group] = new_param - old_param
        
        # Compute pairwise angles
        with torch.no_grad():
            angles = torch.zeros([nb_clients, nb_clients])
            for idc_1, client1 in enumerate(clients):
                for idc_2, client2 in enumerate(clients):
                    
                    c1 = flatten(updates[client1])
                    c2 = flatten(updates[client2])
                    print(c1.shape)
                    angles[idc_1,idc_2] = torch.sum(c1*c2) / (torch.norm(c1)*torch.norm(c2)+1e-12)
                
        angles = angles.detach().numpy()
        print("\nAngles: ", angles)
        
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-angles)

        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten() 
        
        print(c1, c2)
        
        list_clients1 = tuple([clients[c] for c in c1])
        list_clients2 = tuple([clients[c] for c in c2])
        
        print(list_clients1, list_clients2)
        
        sys.exit(0)
        
        
        
    else:
    
        if config["plain_gradient_clustering"]:
            sample_grad = flatten([torch.zeros_like(param) for param in model.parameters()]).detach().cpu().numpy()
            gradient_list = np.zeros((len(ds_city)+len(ds_gta), sample_grad.size), dtype=np.float32)
            print(gradient_list.shape, sample_grad.shape)
            class_distribution = np.zeros((len(ds_city)+len(ds_gta), len(classes)), dtype=int)
            
            if config["labels"] == "trainId":
                loss_function = nn.CrossEntropyLoss(ignore_index=255)
            elif config["labels"] == "categories":
                loss_function = nn.CrossEntropyLoss(ignore_index=255)
                
            useless_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
            # Compute normalized gradients
            for idl, loader in enumerate([loader_city, loader_gta]):
                for idx, batch_data in enumerate(tqdm(loader)):
                        
                    inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
                    sample_grad = [torch.zeros_like(param) for param in model.parameters()]
                    
                    useless_opt.zero_grad()
                    outputs = model(inputs)
                    if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
                        outputs = resize(outputs, size=labels.shape[-2:])
                    loss = loss_function(outputs, labels.squeeze(1))
                    loss.backward()
                    
                    for id_group, param in enumerate(model.parameters()):
                        sample_grad[id_group] = param.grad
                        
                    vect_raw = flatten(sample_grad)
                    vect = normalize(vect_raw, dim=0).detach().cpu().numpy()
                    gradient_list[idl*dataset_size+idx] = vect
                  
                    for c in classes:
                        class_distribution[idl*dataset_size+idx][c] = (labels == c).sum().item()
                        
            # Cluster gradients
            data = gradient_list
            true_label = data_origin
    
            if config["dimension_reduction"] == "std_pruning":
                data, selected_parameters = std_parameter_selection(data, model, config, log_dir, "plain_gradient")
            
            elif config["dimension_reduction"] == "norm_std_pruning":
                data, selected_parameters = norm_std_parameter_selection(data, model, config, log_dir, "plain_gradient")
                
            elif config["dimension_reduction"] == "random":
                data, selected_parameters = random_parameter_selection(data, model, config, log_dir, "plain_gradient")
                
            elif config["dimension_reduction"] == "pca":
                data = pca_reduction(data, config)
                
            elif config["dimension_reduction"] == "TruncatedSVD":
                data = truncatedsvd_reduction(data, config)
                
            if config["clustering"] == "GMM":
                print("Start GMM computation")
                gmm = GaussianMixture(config["nb_clusters"], covariance_type=config["cov_type"], 
                                          random_state=config["seed"], verbose=2, init_params=config["init_gmm"]).fit(data)
                predicted_proba = gmm.predict_proba(data)
                print(predicted_proba.shape)
                predicted_labels = predicted_proba.argmax(axis=-1)
                
                with open(os.path.join(log_dir, f'gmm_plain_gradient.pkl'),'wb') as f:
                    pickle.dump(gmm,f)
                
            elif config["clustering"] == "Agglomerative":
                print("\nAgglomerative clustering")
                clustering = AgglomerativeClustering(n_clusters=config["nb_clusters"], linkage=config["linkage"])
                clustering.fit(data)
                predicted_labels = clustering.labels_
                
            silhouette = silhouette_score(data, predicted_labels)
            rand_ind = rand_score(true_label, predicted_labels)
            print("Silhouette score: ", silhouette, ", Rand index: ", rand_ind)
            
            pd.crosstab(true_label, predicted_labels).to_csv(os.path.join(log_dir, f"{config['clustering']}_plain_gradient_confusion_matrix.csv"))
            pd.DataFrame({"silhouette":silhouette, "rand_index":rand_ind}, index=[0]).to_csv(os.path.join(log_dir, f"{config['clustering']}_plain_gradient_scores.csv"))
           
            # Save clustering with paths of images
            clustered_image_paths = {}
            clustered_target_paths = {}
            for cluster in range(config["nb_spectral_cluster"]):
                
                indices_city = predicted_labels[:dataset_size]
                indices_gta = predicted_labels[dataset_size:]
                
                clustered_image_paths[cluster] = [e for e, good_cluster in zip(ds_city.image_paths, indices_city == cluster) if good_cluster]\
                                                    + [e for e, good_cluster in zip(ds_gta.image_paths, indices_gta == cluster) if good_cluster]
                clustered_target_paths[cluster] = [e for e, good_cluster in zip(ds_city.target_paths, indices_city == cluster) if good_cluster]\
                                                    + [e for e, good_cluster in zip(ds_gta.target_paths, indices_gta == cluster) if good_cluster]
                
            with open(os.path.join(log_dir, "plain_grad_clustered_image_paths.json"), "w") as fp:
                json.dump(clustered_image_paths, fp, indent=4)
            with open(os.path.join(log_dir, "plain_grad_clustered_target_paths.json"), "w") as fp:
                json.dump(clustered_target_paths, fp, indent=4)
                
            # Save clustering with dataset sorte locations
            clustered_numbers = {}
            for cluster in range(config["nb_spectral_cluster"]):
                
                indices_city = predicted_labels[:dataset_size]
                indices_gta = predicted_labels[dataset_size:]
                clustered_numbers[cluster] = {}
                clustered_numbers[cluster]["Cityscapes"] = [e for e, good_cluster in enumerate(indices_city == cluster) if good_cluster]
                clustered_numbers[cluster]["GTA"] = [e for e, good_cluster in enumerate(indices_gta == cluster) if good_cluster]
    
            with open(os.path.join(log_dir, "plain_grad_clustered_numbers.json"), "w") as fp:
                json.dump(clustered_numbers, fp, indent=4)
    
            plot_cluster_class_distribution(predicted_labels, classes, class_distribution, log_dir)
            
            origin_vs_cluster_pacmap(data, true_label, predicted_labels, config, log_dir, "plain_gradient")
            
            
            
            
        # Per class clustering
        if config["per_class"]:
            
            sample_grad = flatten([torch.zeros_like(param) for param in model.parameters()]).detach().cpu().numpy()
            gradient_list = np.zeros((len(ds_city)+len(ds_gta), sample_grad.size), dtype=np.float32)
            print(gradient_list.shape, sample_grad.shape)
            
            if config["labels"] == "trainId":
                loss_per_class = [nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0. if i!=c else 1. for i in range(out_channels)]).to(device)) for c in range(19)]
            elif config["labels"] == "categories":
                loss_per_class = [nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0. if i!=c else 1. for i in range(7)]).to(device)) for c in range(7)]
            
            gmms = {}
            gaussian_weights_per_sample_class = np.zeros((len(ds_city)+len(ds_gta), config["nb_clusters"]*len(classes)))
            
            useless_opt = torch.optim.SGD(model.parameters(), lr=0.1)
                
            for idc, c in enumerate(classes):
                    
                # Compute normalized gradients
                for idl, loader in enumerate([loader_city, loader_gta]):
                    for idx, batch_data in enumerate(tqdm(loader)):
        
                        inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
                        
                        if (labels == c).sum() > 0:
                            
                            sample_grad = [torch.zeros_like(param) for param in model.parameters()]
                            
                            useless_opt.zero_grad()
                            outputs = model(inputs)
                            if config["model"] == "EfficientVIT" and outputs.shape[-2:] != labels.shape[-2:]:
                                outputs = resize(outputs, size=labels.shape[-2:])
                            loss = loss_per_class[c](outputs, labels.squeeze(1))
                            loss.backward()
                            
                            for id_group, param in enumerate(model.parameters()):
                                sample_grad[id_group] = param.grad
                                
                            vect_raw = flatten(sample_grad)
                            
                            if config["normalize_gradients"]:
                                vect = normalize(vect_raw, dim=0).detach().cpu().numpy()
                                gradient_list[len(gradient_list_per_class[c]["mapping"])] = vect
                            else:
                                gradient_list[len(gradient_list_per_class[c]["mapping"])] = vect_raw.detach().cpu().numpy()
                            gradient_list_per_class[c]["mapping"].append(idx+idl*dataset_size)
                 
                print(gradient_list.shape, len(gradient_list_per_class[c]["mapping"]))
                  
                # Cluster gradients
                data = gradient_list[:len(gradient_list_per_class[c]["mapping"]),:]
                true_label = data_origin[gradient_list_per_class[c]["mapping"]]
                
                if config["dimension_reduction"] == "std_pruning":
                    data, selected_parameters = std_parameter_selection(data, model, config, log_dir, f"per_class_clustering/class_{c}")
                    
                elif config["dimension_reduction"] == "norm_std_pruning":
                    data, selected_parameters = norm_std_parameter_selection(data, model, config, log_dir, f"per_class_clustering/class_{c}")
                    
                elif config["dimension_reduction"] == "random":
                    data, selected_parameters = random_parameter_selection(data, model, config, log_dir, f"per_class_clustering/class_{c}")
                    
                if config["dimension_reduction"] == "pca":
                    data = pca_reduction(data, config)
                    
                elif config["dimension_reduction"] == "TruncatedSVD":
                    data = truncatedsvd_reduction(data, config)
                
                if config["clustering"] == "GMM":
                    print("Start GMM computation")
                    gmms[c] = GaussianMixture(config["nb_clusters"], covariance_type=config["cov_type"], 
                                              random_state=config["seed"], verbose=2, init_params=config["init_gmm"]).fit(data)
                    predicted_proba = gmms[c].predict_proba(data)
                    print(predicted_proba.shape)
                    predicted_labels = predicted_proba.argmax(axis=-1)
                    
                    with open(os.path.join(log_dir, "per_class_clustering", f'gmm_{c}.pkl'),'wb') as f:
                        pickle.dump(gmms[c],f)
                    
                elif config["clustering"] == "Agglomerative":
                    print("\nAgglomerative clustering")
                    clustering = AgglomerativeClustering(n_clusters=config["nb_clusters"], linkage=config["linkage"])
                    clustering.fit(data)
                    predicted_labels = clustering.labels_
                    
                silhouette = silhouette_score(data, predicted_labels)
                rand_ind = rand_score(true_label, predicted_labels)
                print("Silhouette score: ", silhouette, ", Rand index: ", rand_ind)
                
                pd.crosstab(true_label, predicted_labels).to_csv(os.path.join(log_dir, "per_class_clustering", f"{config['clustering']}_class_{c}_confusion_matrix.csv"))
                pd.DataFrame({"silhouette":silhouette, "rand_index":rand_ind, "nb_sample_of_class":len(data), "nb_city_samples":(true_label == 0).sum()}, index=[0]).to_csv(os.path.join(log_dir, "per_class_clustering", f"{config['clustering']}_class_{c}_scores.csv"))
                
                # Append to the matrix of gaussian membership 
                for proba, index_in_dataset in zip(predicted_proba, gradient_list_per_class[c]["mapping"]):
                    gaussian_weights_per_sample_class[index_in_dataset, idc*config["nb_clusters"]:(idc + 1)*config["nb_clusters"]] = proba.copy()
                
                origin_vs_cluster_pacmap(data, true_label, predicted_labels, config, log_dir, f"per_class_clustering/class_{c}")
                
                
                
                
            # Spectral clustering on similarity matrix using GMM results per class
            affinity_matrix = np.array([[similarity_gmm(x, y, classes, config) for y in gaussian_weights_per_sample_class] for x in gaussian_weights_per_sample_class])
            
            pd.DataFrame(gaussian_weights_per_sample_class).to_csv(os.path.join(log_dir, "Gaussian_weights.csv"))
            pd.DataFrame(affinity_matrix).to_csv(os.path.join(log_dir, "Similarity_matrix.csv"))
            
            spectral_cluster = SpectralClustering(n_clusters=config["nb_spectral_cluster"], random_state=config["seed"], affinity='precomputed', verbose=True)
            spectral_cluster.fit(affinity_matrix)
            
            pd.DataFrame(spectral_cluster.labels_).to_csv(os.path.join(log_dir, "Cluster_labels.csv"))
            
            rand_ind = rand_score(data_origin, spectral_cluster.labels_)
            
            pd.crosstab(data_origin, spectral_cluster.labels_).to_csv(os.path.join(log_dir, "spectral_clustering_confusion_matrix.csv"))
            pd.DataFrame({"rand_index":rand_ind}, index=[0]).to_csv(os.path.join(log_dir, "spectral_clustering_rand_index.csv"))
            
            # Save clustering with image paths
            clustered_image_paths = {}
            clustered_target_paths = {}
            for cluster in range(config["nb_spectral_cluster"]):
                
                indices_city = spectral_cluster.labels_[:dataset_size]
                indices_gta = spectral_cluster.labels_[dataset_size:]
                
                clustered_image_paths[cluster] = [e for e, good_cluster in zip(ds_city.image_paths, indices_city == cluster) if good_cluster]\
                                                    + [e for e, good_cluster in zip(ds_gta.image_paths, indices_gta == cluster) if good_cluster]
                clustered_target_paths[cluster] = [e for e, good_cluster in zip(ds_city.target_paths, indices_city == cluster) if good_cluster]\
                                                    + [e for e, good_cluster in zip(ds_gta.target_paths, indices_gta == cluster) if good_cluster]
                
            with open(os.path.join(log_dir, "clustered_image_paths.json"), "w") as fp:
                json.dump(clustered_image_paths, fp, indent=4)
            with open(os.path.join(log_dir, "clustered_target_paths.json"), "w") as fp:
                json.dump(clustered_target_paths, fp, indent=4)
        
            # Save clustering with dataset sorte locations
            clustered_numbers = {}
            for cluster in range(config["nb_spectral_cluster"]):
                
                indices_city = spectral_cluster.labels_[:dataset_size]
                indices_gta = spectral_cluster.labels_[dataset_size:]
                clustered_numbers[cluster] = {}
                clustered_numbers[cluster]["Cityscapes"] = [e for e, good_cluster in enumerate(indices_city == cluster) if good_cluster]
                clustered_numbers[cluster]["GTA"] = [e for e, good_cluster in enumerate(indices_gta == cluster) if good_cluster]
    
            with open(os.path.join(log_dir, "clustered_numbers.json"), "w") as fp:
                json.dump(clustered_numbers, fp, indent=4)

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