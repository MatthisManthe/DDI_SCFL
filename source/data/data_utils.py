import os
import numpy as np
import glob
import pandas as pd


def merge_inst_cluster_part(institution_part_image, institution_part_label, cluster_part_image):
    
    """ 
    cluster_part: {cluster:list of samples}
    institution_part: {institution:list of samples}
    
    result_part: {cluster:{institution:list of samples}}
    """
    
    cluster_part_image_base = {cluster:[os.path.basename(image) for image in images] for cluster, images in cluster_part_image.items()}
    institution_part_image_base = {client:[os.path.basename(image) for image in images] for client, images in institution_part_image.items()}
    
    result_part_image = {}
    result_part_label = {}
    
    for (client, base_images), full_images, full_labels in zip(institution_part_image_base.items(), institution_part_image.values(), institution_part_label.values()):
        
        for base_image, full_image, full_label in zip(base_images, full_images, full_labels):
            print(base_image, full_image, full_label)
            
            for cluster, samples_cluster in cluster_part_image_base.items():
                
                if base_image in samples_cluster:
                    
                    if cluster not in result_part_image:
                        result_part_image[cluster] = {}
                        result_part_label[cluster] = {}
                    if client in result_part_image[cluster]:
                        result_part_image[cluster][client].append(full_image)
                        result_part_label[cluster][client].append(full_label)
                    else:
                        result_part_image[cluster][client] = [full_image]
                        result_part_label[cluster][client] = [full_label]
             
    return result_part_image, result_part_label


def merge_inst_and_cluster_numbers(institution_numbers, cluster_numbers):
    
    """
    cluster_numbers: {cluster:{origin:list of samples}}
    institution_numbers: {institution:{origin:list of samples}}
    
    result_numbers: {cluster:{institution:{origin:list of samples}}}
    """

    result_numbers = {cluster:{} for cluster in cluster_numbers}
    
    for cluster, cluster_datasets in cluster_numbers.items():
        
        for origin, sample_numbers_origin_cluster in cluster_datasets.items():
            
            for client, local_datasets in institution_numbers.items():
                
                if origin in local_datasets:
    
                    intersect = [value for value in sample_numbers_origin_cluster if value in local_datasets[origin]]
                    
                    if len(intersect) > 0:
             
                        if client not in result_numbers[cluster]:
                            result_numbers[cluster][client] = {origin:intersect.copy()}
                        elif origin not in result_numbers[cluster][client]:
                            result_numbers[cluster][client][origin] = intersect.copy()
                        else:
                            result_numbers[cluster][client][origin].extend(intersect.copy())
       
    return result_numbers

"""
inst = {
    "I1":{
        "O1":[1, 2, 3, 4], 
        "O2":[1, 2, 3, 4]
        
    },
    "I2":{
        "O1":[5, 6, 7, 8],
        "O2":[5, 6, 7, 8]
    }  
}

cluster = {
    "C1":{
        "O1":[1, 2, 5, 6],
        "O2":[1, 2, 5, 6]
    },
    "C2":{
        "O1":[3, 4, 7, 8],
        "O2":[3, 4, 7, 8]
    }
}
from pprintpp import pprint
pprint(merge_inst_and_cluster_numbers(inst, cluster))

cluster = {
    "C1":{
        "O1":[1, 2, 3, 4, 5, 6, 7, 8]
    },
    "C2":{
        "O2":[1, 2, 3, 4, 5, 6, 7, 8]
    }
}
pprint(merge_inst_and_cluster_numbers(inst, cluster))
"""

def recursive_add_parent_in_path(path_dict):
    
    if isinstance(path_dict, dict):
        for key, value in path_dict.items():
            path_dict[key] = recursive_add_parent_in_path(value)
    elif isinstance(path_dict, list):
        for i, element in enumerate(path_dict):
            path_dict[i] = recursive_add_parent_in_path(element)
    elif isinstance(path_dict, str):
        path_dict = "../"+path_dict
    
    return path_dict

class DataLoaderWithMemory:
    """This class allows to iterate the dataloader infinitely batch by batch.
    When there are no more batches the iterator is reset silently.
    This class allows to keep the memory of the state of the iterator hence its
    name.
    """

    def __init__(self, dataloader):
        """This initialization takes a dataloader and creates an iterator object
        from it.
        Parameters
        ----------
        dataloader : torch.utils.data.dataloader
            A dataloader object built from one of the datasets of this repository.
        """
        self._dataloader = dataloader

        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader.dataset)

    def get_samples(self):
        """This method generates the next batch from the iterator or resets it
        if needed. It can be called an infinite amount of times.
        Returns
        -------
        tuple
            a batch from the iterator
        """
        try:
            X = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            X = next(self._iterator)
        return X