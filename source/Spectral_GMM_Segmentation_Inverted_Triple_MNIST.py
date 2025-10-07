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

import matplotlib.lines as mlines
import matplotlib

from sklearn.metrics import classification_report, multilabel_confusion_matrix
from data.datasets import TripleMNISTSegmentationDataset
from models.unet import UNet
from models.models import FCN_Triple_MNIST_Segmentation

gaussian_weights = None
similarity_mat = None

test_data_interactive = None
output_masks_interactive = None
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


def onpick_IT_MNIST(event):
    
    global test_data_interactive, output_masks_interactive, id_color, colors

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
    fig, axs = plt.subplots(row, col*3, squeeze=False, figsize=(15,15))
    for dataind, ax_id in zip(event.ind, range(0, len(event.ind)*3, 3)):
        axs[ax_id//(col*3)][ax_id%(col*3)].imshow(test_data_interactive[dataind][0][0], cmap="gray", vmin=0, vmax=1)
        axs[ax_id//(col*3)][ax_id%(col*3)+1].imshow(test_data_interactive[dataind][1], vmin=0, vmax=10, cmap="hot")
        axs[ax_id//(col*3)][ax_id%(col*3)+2].imshow(output_masks_interactive[dataind], vmin=0, vmax=10, cmap="hot")
        
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
    
    if config["explore_embedding_per_class"]:
        fig.canvas.mpl_connect('pick_event', onpick_IT_MNIST)
        plt.show()
    else: 
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
    class_space = 10
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


def invert_full_dataset(dataset):
    return 255 - dataset


def main(args, config):
    """Main function"""
    
    global test_data_interactive, output_masks_interactive
    
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
    
    train_images = np.load(os.path.join(config["dataset"], config["split"], config["image_path"]))
    train_masks = np.load(os.path.join(config["dataset"], config["split"], "masks.npy"))          
    
    # Half of the images of the datasets are altered
    train_images_clear = train_images[:len(train_images)//2]
    train_masks_clear = train_masks[:len(train_images)//2]
    train_images_inverted = train_images[len(train_images)//2:]
    train_masks_inverted = train_masks[len(train_images)//2:]
        
    if config["flip_horizontal"]:
        # Half of the images of the datasets are flipped
        train_images_inverted = np.flip(train_images_inverted, axis=-1)
        train_masks_inverted = np.flip(train_masks_inverted, axis=-1)
        
    if config["flip_vertical"]:
        # Half of the images of the datasets are flipped
        train_images_inverted = np.flip(train_images_inverted, axis=-2)
        train_masks_inverted = np.flip(train_masks_inverted, axis=-2)
        
    if config["flip_90"]:
        train_images_clear = np.pad(train_images_clear, ((0,0), (16, 16), (0,0)), constant_values=0)
        train_masks_clear = np.pad(train_masks_clear, ((0,0), (16, 16), (0,0)), constant_values=0)
        train_images_inverted = np.pad(train_images_inverted, ((0,0), (16, 16), (0,0)), constant_values=0)
        train_masks_inverted = np.pad(train_masks_inverted, ((0,0), (16, 16), (0,0)), constant_values=0)
        
        train_images_inverted = np.rot90(train_images_inverted, axes=(-2, -1))
        train_masks_inverted = np.rot90(train_masks_inverted, axes=(-2, -1))
    
    if config["invert"]:
        train_images_inverted = invert_full_dataset(train_images_inverted)
        train_masks_inverted = train_masks_inverted
        
    if config["subset"] < 1:
        train_images_clear = train_images_clear[:int(len(train_images_clear)*config["subset"])]
        train_masks_clear = train_masks_clear[:int(len(train_masks_clear)*config["subset"])]
        train_images_inverted = train_images_inverted[:int(len(train_images_inverted)*config["subset"])]
        train_masks_inverted = train_masks_inverted[:int(len(train_masks_inverted)*config["subset"])]
        
    print(train_images_inverted.shape, train_images_clear.shape, train_masks.shape)
    
    train_dataset = TripleMNISTSegmentationDataset(
        np.concatenate((train_images_clear, train_images_inverted), axis=0),
        np.concatenate((train_masks_clear, train_masks_inverted), axis=0)    
    )
    train_data_clear = torch.utils.data.Subset(train_dataset, list(range(len(train_images_clear))))
    train_data_inverted = torch.utils.data.Subset(train_dataset, list(range(len(train_images_clear), len(train_images_clear)+len(train_images_inverted))))

    print(train_data_clear)
    print(len(train_data_clear))
    plt.imshow(train_data_clear[0][0][0], cmap="gray")
    plt.title(train_data_clear[0][1])
    plt.show()
    plt.imshow(train_data_clear[1][0][0], cmap="gray")
    plt.title(train_data_clear[1][1])
    plt.show() 
    plt.imshow(train_data_inverted[0][0][0], cmap="gray")
    plt.title(train_data_inverted[0][1])
    plt.show()
    
    print(train_data_clear[0][0].min(), train_data_clear[0][0].max())
    print(train_data_inverted[0][0].min(), train_data_inverted[0][0].max())
    
    train_loader_clear = torch.utils.data.DataLoader(train_data_clear, batch_size=1, shuffle=False, num_workers=0)
    train_loader_inverted = torch.utils.data.DataLoader(train_data_inverted, batch_size=1, shuffle=False, num_workers=0)

    print(next(iter(train_loader_clear))[0].shape, next(iter(train_loader_clear))[1].shape)
    print(next(iter(train_loader_inverted))[0].shape, next(iter(train_loader_inverted))[1].shape)
    
    if len(config["digits"]) > 0:
        nb_classes = len(config["digits"])+1
        classes = list(range(nb_classes))
        classes_name = ["Background"]+[str(d) for d in config["digits"]]
        
    else:
        nb_classes = 11
        classes = list(range(nb_classes))
        classes_name = ["Background"]+list("0123456789")

    if config["model"] == "My_fcn":
        model = FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device)
        local_model = FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device)
    else:
        model = UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 11,
            kernel_size = config["kernel_sizes"],
            features = config["filters"],
            strides = config["strides"],
            dropout=0
        ).to(device)
        local_model = UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 11,
            kernel_size = config["kernel_sizes"],
            features = config["filters"],
            strides = config["strides"],
            dropout=0
        ).to(device)

    print(model)
    print(summary(model, (1,96,96)))
    
    if not config["rand_init"]:
        model.load_state_dict(torch.load(os.path.join(config["model_dir"], config["trained_model_file"])))
        
    data_origin = np.array([0]*len(train_data_clear)+[1]*len(train_data_inverted))
    print("Data_origin: ", data_origin)  
    
    if config["subset_classes"] != 0:
        classes = classes[:config["subset_classes"]]
        
    os.mkdir(os.path.join(log_dir, "per_class_clustering"))
    
    # Compute class-wise average gradients
    gradient_list_per_class = {c:{"name": c, "gradients":np.array([]), "mapping":[]} for c in classes}
      
    model.train()
    
    if config["clustering"] == "CFL":
        
        with open(os.path.join(config["partitioning_file"]), 'r') as f:
            local_samples = json.load(f)
            
        clients = list(local_samples.keys())
        nb_clients = len(clients)
        dataset_size = len(train_data_clear)
        
        local_datasets = {} 
        for c in clients:
            full_local_samples = []
            if "Clear" in local_samples[c]:
                full_local_samples.extend(local_samples[c]["Clear"])
            if "Inverted" in  local_samples[c]:
                full_local_samples.extend([ids+dataset_size for ids in local_samples[c]["Inverted"]])
            local_datasets[c] = torch.utils.data.Subset(train_dataset, full_local_samples)
            
        local_loaders = {c:torch.utils.data.DataLoader(local_datasets[c], batch_size=config["batch_size"], shuffle=True, num_workers=0) for c in clients}
        
        weights = torch.tensor([0.01] + [0.99/(nb_classes-1)]*(nb_classes-1)).to(device)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        learning_rate = config["learning_rate"]*(config["gamma"]**config["split_round"])
        print(learning_rate)
        local_optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        updates = {client:[torch.zeros_like(param) for param in model.parameters()] for client in clients}
        
        for client, loader in local_loaders.items():
            for local_param, global_param in zip(local_model.parameters(), model.parameters()):
                local_param.data = global_param.data.clone()
                
            local_model.train()
            
            for i, (images, labels) in enumerate(tqdm(loader)):
    
                output = local_model(images.to(device))     
                loss = loss_func(output, labels.to(device))
                
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
                
    # Plain gradient clustering, see what it does (should not be perfect, affected by class distribution in images)
    else:
        if config["plain_gradient_clustering"]:
            sample_grad = flatten([torch.zeros_like(param) for param in model.parameters()]).detach().cpu().numpy()
            gradient_list = np.zeros((len(train_data_clear)+len(train_data_inverted), sample_grad.size), dtype=np.float32)
            print(gradient_list.shape, sample_grad.shape)
            class_distribution = np.zeros((len(train_data_clear)+len(train_data_inverted), len(classes)), dtype=int)
            
            weights = torch.tensor([0.01] + [0.99/(nb_classes-1)]*(nb_classes-1)).to(device)
            loss_function = nn.CrossEntropyLoss(weight=weights)
            
            useless_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
            # Compute normalized gradients
            for idl, loader in enumerate([train_loader_clear, train_loader_inverted]):
                for idx, batch_data in enumerate(tqdm(loader)):
                        
                    inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
                    sample_grad = [torch.zeros_like(param) for param in model.parameters()]
                    
                    useless_opt.zero_grad()
                    outputs = model(inputs)
    
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    
                    for id_group, param in enumerate(model.parameters()):
                        sample_grad[id_group] = param.grad
                        
                    vect_raw = flatten(sample_grad)
                    vect = normalize(vect_raw, dim=0).detach().cpu().numpy()
                    gradient_list[idl*len(train_data_clear)+idx] = vect
                    
                    for c in classes:
                        class_distribution[idl*len(train_data_clear)+idx][c] = (labels == c).sum().item()
                    
                        
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
                                          random_state=config["seed"], verbose=2, init_params=config["init_gmm"], verbose_interval=1).fit(data)
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
    
            if config["plot_some_examples_per_cluster"]:
                for c in range(config["nb_clusters"]):
                    nb_samples = 0
                    id_clear = 0
                    while nb_samples < 20:
                        if predicted_labels[id_clear] == c:
                            plt.figure(figsize=(10, 10))
                            plt.imshow(train_images_clear[id_clear])
                            plt.imshow(train_masks_clear[id_clear], cmap="hot", alpha=0.3, vmin=0, vmax=10)
                            plt.title(f"Cluster {c}")
                            plt.show()
                            nb_samples += 1
                        id_clear += 1
                       
                    nb_samples = 0
                    id_inverted = 0
                    while nb_samples < 20:
                        if predicted_labels[len(train_data_clear)+id_inverted] == c:
                            plt.figure(figsize=(10, 10))
                            plt.imshow(train_images_inverted[id_inverted])
                            plt.imshow(train_masks_inverted[id_inverted], cmap="hot", alpha=0.3, vmin=0, vmax=10)
                            plt.title(f"Cluster {c}")
                            plt.show()
                            nb_samples += 1
                        id_inverted += 1
    
            # Save clustering with dataset sorted locations
            clustered_numbers = {}
            for cluster in range(config["nb_spectral_cluster"]):
                
                indices_clear = predicted_labels[:len(train_data_clear)]
                indices_inverted = predicted_labels[len(train_data_clear):]
                clustered_numbers[cluster] = {}
                clustered_numbers[cluster]["Clear"] = [e for e, good_cluster in enumerate(indices_clear == cluster) if good_cluster]
                clustered_numbers[cluster]["Inverted"] = [e for e, good_cluster in enumerate(indices_inverted == cluster) if good_cluster]
    
            with open(os.path.join(log_dir, "plain_grad_clustered_numbers.json"), "w") as fp:
                json.dump(clustered_numbers, fp, indent=4)
    
            plot_cluster_class_distribution(predicted_labels, classes, class_distribution, log_dir)
            
            origin_vs_cluster_pacmap(data, true_label, predicted_labels, config, log_dir, "plain_gradient")
            
            
        # Per class clustering
        if config["per_class"]:
            
            sample_grad = flatten([torch.zeros_like(param) for param in model.parameters()]).detach().cpu().numpy()
            gradient_list = np.zeros((len(train_data_clear)+len(train_data_inverted), sample_grad.size), dtype=np.float32)
            print(gradient_list.shape, sample_grad.shape)
            
            #loss_per_class = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0. if i!=c else 1. for i in range(10)]).to(device)) for c in range(10)]
            loss_per_class = [nn.CrossEntropyLoss(weight=torch.tensor([0. if i!=c else 1. for i in range(nb_classes)]).to(device)) for c in range(nb_classes)]
            
            gmms = {}
            if len(config["nb_clusters_per_class"]) > 0:
                gaussian_weights_per_sample_class = np.zeros((len(train_data_clear)+len(train_data_inverted), np.sum(config["nb_clusters_per_class"])))
            else:
                gaussian_weights_per_sample_class = np.zeros((len(train_data_clear)+len(train_data_inverted), config["nb_clusters"]*len(classes)))
            
            useless_opt = torch.optim.SGD(model.parameters(), lr=0.1)
                
            print(classes)
            
            cum_nb_cluster = 0
            
            for idc, c in enumerate(classes):
                   
                if len(config["nb_clusters_per_class"]) > 0:
                    current_nb_cluster = config["nb_clusters_per_class"][idc]
                else:
                    current_nb_cluster = config["nb_clusters"]
                    
                if config["explore_embedding_per_class"]:
                    output_masks_interactive = []
                    
                # Compute normalized gradients
                for idl, loader in enumerate([train_loader_clear, train_loader_inverted]):
                    for idx, batch_data in enumerate(tqdm(loader)):
        
                        inputs, labels = batch_data[0].to(device), batch_data[1].long().to(device)
                        
                        if (labels == c).sum() > 0:
                            
                            sample_grad = [torch.zeros_like(param) for param in model.parameters()]
                            
                            useless_opt.zero_grad()
                            outputs = model(inputs)
    
                            if config["explore_embedding_per_class"]:
                                output_masks_interactive.append(torch.argmax(outputs, dim=1, keepdim=True).squeeze(0).squeeze(0).detach().cpu().numpy())
                                
                            loss = loss_per_class[c](outputs, labels)
                            loss.backward()
                            
                            for id_group, param in enumerate(model.parameters()):
                                sample_grad[id_group] = param.grad
                                
                            vect_raw = flatten(sample_grad)
                            
                            if config["normalize_gradients"]:
                                vect = normalize(vect_raw, dim=0).detach().cpu().numpy()
                                gradient_list[len(gradient_list_per_class[c]["mapping"])] = vect
                            else:
                                gradient_list[len(gradient_list_per_class[c]["mapping"])] = vect_raw.detach().cpu().numpy()
                            gradient_list_per_class[c]["mapping"].append(idx+idl*len(train_data_clear))
                 
                print(gradient_list.shape, len(gradient_list_per_class[c]["mapping"]))
                  
                # Cluster gradients
                if config["explore_embedding_per_class"]:
                    test_data_interactive = torch.utils.data.Subset(ConcatDataset([train_data_clear, train_data_inverted]), gradient_list_per_class[c]["mapping"])
                data = gradient_list[:len(gradient_list_per_class[c]["mapping"]),:]
                true_label = data_origin[gradient_list_per_class[c]["mapping"]]
                
                print(data.shape, true_label.shape)
                
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
                    gmms[c] = GaussianMixture(current_nb_cluster, covariance_type=config["cov_type"], 
                                              random_state=config["seed"], verbose=2, init_params=config["init_gmm"], verbose_interval=1).fit(data)
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
                    gaussian_weights_per_sample_class[index_in_dataset, cum_nb_cluster:cum_nb_cluster+current_nb_cluster] = proba.copy()
                
                origin_vs_cluster_pacmap(data, true_label, predicted_labels, config, log_dir, f"per_class_clustering/class_{c}")
                
                cum_nb_cluster += current_nb_cluster
                
                
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
            
            # Save clustering with dataset sorte locations
            clustered_numbers = {}
            for cluster in range(config["nb_spectral_cluster"]):
                
                indices_clear = spectral_cluster.labels_[:len(train_data_clear)]
                indices_inverted = spectral_cluster.labels_[len(train_data_clear):]
                clustered_numbers[cluster] = {}
                clustered_numbers[cluster]["Clear"] = [e for e, good_cluster in enumerate(indices_clear == cluster) if good_cluster]
                clustered_numbers[cluster]["Inverted"] = [e for e, good_cluster in enumerate(indices_inverted == cluster) if good_cluster]
    
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