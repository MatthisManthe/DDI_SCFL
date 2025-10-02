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
from data.data_utils import recursive_add_parent_in_path, merge_inst_cluster_part
from models.models import CNN_GTA_Cityscapes_source_Classification, Pool_CNN_GTA_Cityscapes_source_Classification

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
        plt.title(f"Cityscapes (label {train_ds_city[i][1][0].item()})")
        plt.tight_layout()
        writer.add_figure(f"Verification_Cityscapes", fig, global_step=i)
        if i > 10:
            break
    for i in range(len(train_ds_gta)):
        fig, axs = plt.subplots(1, 2, figsize=[12,12])
        axs[0].imshow(prepare_plot_im(train_ds_gta[i][0]))
        plt.title(f"GTA5 (label {train_ds_gta[i][1][0].item()})")
        plt.tight_layout()
        writer.add_figure(f"Verification_GTA", fig, global_step=i)
        if i > 10:
            break

    if config["prop_full_dataset"] < 1:
        train_ds_city.subset(list(range(int(len(train_ds_city)*config["prop_full_dataset"]))))
        train_ds_gta.subset(list(range(int(len(train_ds_gta)*config["prop_full_dataset"]))))
        val_ds_city.subset(list(range(int(len(val_ds_city)*config["prop_full_dataset"]))))
        val_ds_gta.subset(list(range(int(len(val_ds_gta)*config["prop_full_dataset"]))))
        
    train_ds = ConcatDataset([train_ds_city, train_ds_gta])
    val_ds = ConcatDataset([val_ds_city, val_ds_gta])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    
    class_names = ["Cityscapes", "GTA"]
    
    print("Train size: ", len(train_ds))
    print("Val size: ", len(val_ds))

    pprint(train_ds)
    
    im, labels = next(iter(train_loader))
    print(im.shape, labels.shape, im.min(), im.max(), im.mean(), im.std())
        
    device = torch.device("cuda:0")
    
    model = CNN_GTA_Cityscapes_source_Classification(sources=2).to(device)
        
    print(model)
        
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    
    # Adding graph of model to tensorboard and print it, and specific instanciations for specific datasets
    writer.add_graph(model, next(iter(train_loader))[0].to(device))
    print(next(iter(train_loader))[0].size())
    summary(model, next(iter(train_loader))[0][0].size())
    
    # Get losses, metrics and optimizers
    loss_function = nn.CrossEntropyLoss()
    best_metric = 0.0
  
    for epoch in range(config["max_epochs"]):  # loop over the dataset multiple times

        epoch_loss = 0.0
        
        model.train()
        
        writer.add_scalar("Learning rate/Epoch", optimizer.param_groups[0]['lr'], epoch+1)
        
        for idx, batch_data in enumerate(tqdm(train_loader)):

            inputs, labels = batch_data[0].to(device), batch_data[1].to(device).long()
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            output = model(inputs)     
            loss = loss_function(output, labels.squeeze(1))
        
            loss.backward()
                
            optimizer.step()
    
            # print statistics
            epoch_loss += loss.item()*batch_size
    
        epoch_loss /= len(train_ds)
    
        # Add loss value to tensorboard, and print it
        writer.add_scalar(f"Loss/train", epoch_loss, epoch+1)
        print(f"\nEpoch {epoch + 1}, average loss: {epoch_loss:.4f}")
        
        # Validation step
        if (epoch) % config["validation_interval"] == 0:
            
            model.eval()
            
            with torch.no_grad():
                
                # Main loop on validation set
                for (idx, batch_data) in enumerate(tqdm(val_loader)):
                    
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    test_output = model(inputs)
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
                    writer.add_scalar(f"Validation/F1_score_class_{cl}", report_dict[cl]["f1-score"], epoch + 1)
                writer.add_scalar("Validation/Weighted_average_F1_score", report_dict["weighted avg"]["f1-score"], epoch + 1)
                
                print(report)
                
                if best_metric < report_dict["weighted avg"]["f1-score"]:
                    best_metric = report_dict["weighted avg"]["f1-score"]
                    torch.save(model.state_dict(), os.path.join(log_dir, "model.pth"))
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
