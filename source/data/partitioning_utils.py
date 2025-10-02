import os
import numpy as np
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt
import sys



# -------------------------- GTA5 + Cityscapes partitioning utils -------------------------


def GTA_Cityscapes_iid(dataset_size, clients, save_folder=None, seed=0, split="train"):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    nb_clients = len(clients)
    final_assignement = {clients[c]:{"Cityscapes":permut_1[int(c/nb_clients*dataset_size):int((c+1)/nb_clients*dataset_size)].tolist(),
                                     "GTA":permut_2[int(c/nb_clients*dataset_size):int((c+1)/nb_clients*dataset_size)].tolist()} 
                         for c in range(nb_clients)}
    
    if save_folder is not None:
        with open(os.path.join(save_folder, f'{split}_partitioning_GTA_Cityscapes.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
    
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Cityscapes"]) for c in clients], label="Cityscapes")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["GTA"]) for c in clients], bottom=[len(final_assignement[c]["Cityscapes"]) for c in clients], label="GTA5")
    plt.title("Number of samples from Cityscapes and GTA5 dataset per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"Data_origin_distribution_per_client_{split}.svg"))
    
    return final_assignement


def GTA_Cityscapes_Full_non_iid(dataset_size, clients, save_folder=None, seed=0, split="train"):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    nb_clients_per_set = int(len(clients)/2.)
    
    final_assignement = {clients[c]:{"Cityscapes":permut_1[int(c/nb_clients_per_set*dataset_size):int((c+1)/nb_clients_per_set*dataset_size)].tolist()} for c in range(nb_clients_per_set)}
    final_assignement.update({clients[c+nb_clients_per_set]:{"GTA":permut_2[int(c/nb_clients_per_set*dataset_size):int((c+1)/nb_clients_per_set*dataset_size)].tolist()} for c in range(nb_clients_per_set)})
    
    print(final_assignement)
    
    if save_folder is not None:
        with open(os.path.join(save_folder, f'{split}_partitioning_GTA_Cityscapes.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
      
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Cityscapes"]) if "Cityscapes" in final_assignement[c] else 0 for c in clients], label="Cityscapes")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["GTA"]) if "GTA" in final_assignement[c] else 0 for c in clients], bottom=[len(final_assignement[c]["Cityscapes"]) if "Cityscapes" in final_assignement[c] else 0 for c in clients], label="GTA5")
    plt.title("Number of samples from Cityscapes and GTA5 dataset per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"Data_origin_distribution_per_client_{split}.svg"))
    
    return final_assignement


def GTA_Cityscapes_Dirichlet_non_iid(dataset_size, clients, dirichlet, save_folder=None, seed=0, split="train"):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    # With rough dirichlet per client, avoid having two distinct distributions in the federation
    total_size = dataset_size*2
    
    np.random.seed(seed)
    client_distrib = np.random.dirichlet(np.ones(2)*dirichlet, size=len(clients))
    print(client_distrib)
    
    max_samples = total_size/len(clients)
    
    nb_samples_per_origin_and_client_ndarray = (client_distrib*max_samples).round().astype(int)
    for row in range(len(clients)):
        while np.sum(nb_samples_per_origin_and_client_ndarray[row]) > max_samples:
            down_class = np.argmax(nb_samples_per_origin_and_client_ndarray[row])
            nb_samples_per_origin_and_client_ndarray[row, down_class] = nb_samples_per_origin_and_client_ndarray[row, down_class] - 1
         
    print(client_distrib.shape, client_distrib.sum(axis=0))
    print(nb_samples_per_origin_and_client_ndarray, nb_samples_per_origin_and_client_ndarray.sum(axis=0), nb_samples_per_origin_and_client_ndarray.sum(axis=1))
    
    origin_with_to_many_samples = np.argmax(nb_samples_per_origin_and_client_ndarray.sum(axis=0))
    
    while nb_samples_per_origin_and_client_ndarray.sum(axis=0)[0] != nb_samples_per_origin_and_client_ndarray.sum(axis=0)[1]:
        
        client_to_equilibrate = np.random.choice(np.arange(len(clients))[nb_samples_per_origin_and_client_ndarray[:,origin_with_to_many_samples] > 0])
        
        nb_samples_per_origin_and_client_ndarray[client_to_equilibrate, origin_with_to_many_samples] -= 1
        nb_samples_per_origin_and_client_ndarray[client_to_equilibrate, 1-origin_with_to_many_samples] += 1
        
    print(client_distrib.shape, client_distrib.sum(axis=0))
    print(nb_samples_per_origin_and_client_ndarray, nb_samples_per_origin_and_client_ndarray.sum(axis=0), nb_samples_per_origin_and_client_ndarray.sum(axis=1))

    current_1 = 0
    current_2 = 0
    final_assignement = {}
    for c in range(len(clients)):
        final_assignement[clients[c]] = {}
        if nb_samples_per_origin_and_client_ndarray[c, 0] > 0:
            final_assignement[clients[c]]["Cityscapes"] = permut_1[current_1:current_1+nb_samples_per_origin_and_client_ndarray[c, 0]].tolist()
            current_1 += nb_samples_per_origin_and_client_ndarray[c, 0]
        if nb_samples_per_origin_and_client_ndarray[c, 1] > 0:
            final_assignement[clients[c]]["GTA"] = permut_2[current_2:current_2+nb_samples_per_origin_and_client_ndarray[c, 1]].tolist()
            current_2 += nb_samples_per_origin_and_client_ndarray[c, 1]

    if save_folder is not None:
        with open(os.path.join(save_folder, f'{split}_partitioning_GTA_Cityscapes.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
            
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Cityscapes"]) if "Cityscapes" in final_assignement[c] else 0 for c in clients], label="Cityscapes")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["GTA"]) if "GTA" in final_assignement[c] else 0 for c in clients], bottom=[len(final_assignement[c]["Cityscapes"]) if "Cityscapes" in final_assignement[c] else 0 for c in clients], label="GTA5")
    plt.title("Number of samples from Cityscapes and GTA5 dataset per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"Data_origin_distribution_per_client_{split}.svg"))

    return final_assignement
    

def assert_GTA_Cityscapes_partitioning(final_assignement, clients):
    
    for c1 in clients:
        if "Cityscapes" in final_assignement[c1]:
            for c2 in clients:
                if c1 != c2 and "Cityscapes" in final_assignement[c2]:
                    print(f"{c1} vs {c2} for Cityscapes", end="")
                    assert not bool(set(final_assignement[c1]["Cityscapes"]) & set(final_assignement[c2]["Cityscapes"]))
                    
    for c1 in clients:
        if "GTA" in final_assignement[c1]:
            for c2 in clients:
                if c1 != c2 and "GTA" in final_assignement[c2]:
                    print(f"{c1} vs {c2} for GTA", end="")
                    assert not bool(set(final_assignement[c1]["GTA"]) & set(final_assignement[c2]["GTA"]))
                    
    print("\nClean partitioning")
    
# -------------------------- Triple MNIST partitioning utils ----------------------------- 



def Inverted_Triple_MNIST_iid(dataset_size, clients, save_folder=None, seed=0, split="train"):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    nb_clients = len(clients)
    final_assignement = {clients[c]:{"Clear":permut_1[int(c/nb_clients*dataset_size):int((c+1)/nb_clients*dataset_size)].tolist(),
                                     "Inverted":permut_2[int(c/nb_clients*dataset_size):int((c+1)/nb_clients*dataset_size)].tolist()} 
                         for c in range(nb_clients)}
    
    if save_folder is not None:
        with open(os.path.join(save_folder, f'{split}_partitioning_Inverted_Triple_MNIST.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
    
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Clear"]) for c in clients], label="Clear")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Inverted"]) for c in clients], bottom=[len(final_assignement[c]["Clear"]) for c in clients], label="Inverted")
    plt.title("Number of clear and inverted samples per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"Data_origin_distribution_per_client_{split}.svg"))
    
    return final_assignement


def Inverted_Triple_MNIST_Full_non_iid(dataset_size, clients, save_folder=None, seed=0, split="train"):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    nb_clients_per_set = int(len(clients)/2.)
    
    final_assignement = {clients[c]:{"Clear":permut_1[int(c/nb_clients_per_set*dataset_size):int((c+1)/nb_clients_per_set*dataset_size)].tolist()} for c in range(nb_clients_per_set)}
    final_assignement.update({clients[c+nb_clients_per_set]:{"Inverted":permut_2[int(c/nb_clients_per_set*dataset_size):int((c+1)/nb_clients_per_set*dataset_size)].tolist()} for c in range(nb_clients_per_set)})
    
    print(final_assignement)
    
    if save_folder is not None:
        with open(os.path.join(save_folder, f'{split}_partitioning_Inverted_Triple_MNIST.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
      
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Clear"]) if "Clear" in final_assignement[c] else 0 for c in clients], label="Clear")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Inverted"]) if "Inverted" in final_assignement[c] else 0 for c in clients], bottom=[len(final_assignement[c]["Clear"]) if "Clear" in final_assignement[c] else 0 for c in clients], label="Inverted")
    plt.title("Number of clear and inverted samples per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"Data_origin_distribution_per_client_{split}.svg"))
    
    return final_assignement


def Inverted_Triple_MNIST_Fraction_non_iid(dataset_size, clients, fraction, save_folder=None, seed=0):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    nb_clients_per_group = int(len(clients)/2.)
    
    nb_samples_per_client_per_group = [int(dataset_size*2/len(clients)*fraction) if i < len(clients)//2 else int(dataset_size*2/len(clients)*(1-fraction)) for i in range(len(clients))]
    
    print(nb_samples_per_client_per_group, np.sum(nb_samples_per_client_per_group))
    
    
    final_assignement = {client:{} for client in clients}
    current_1 = 0
    current_2 = 0
    for c in range(len(clients)):
        final_assignement[clients[c]]["Clear"] = permut_1[current_1:current_1+nb_samples_per_client_per_group[c]].tolist()
        final_assignement[clients[c]]["Inverted"] = permut_2[current_2:current_2+nb_samples_per_client_per_group[len(clients)-1-c]].tolist()
        current_1 += nb_samples_per_client_per_group[c]
        current_2 += nb_samples_per_client_per_group[len(clients)-1-c]
        
    print(final_assignement)
    
    if save_folder is not None:
        with open(os.path.join(save_folder, 'train_partitioning_Inverted_Triple_MNIST.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
      
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Clear"]) if "Clear" in final_assignement[c] else 0 for c in clients], label="Clear")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Inverted"]) if "Inverted" in final_assignement[c] else 0 for c in clients], bottom=[len(final_assignement[c]["Clear"]) if "Clear" in final_assignement[c] else 0 for c in clients], label="Inverted")
    plt.title("Number of clear and inverted samples per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "Data_origin_distribution_per_client.svg"))
    
    return final_assignement


def Inverted_Triple_MNIST_Dirichlet_non_iid(dataset_size, clients, dirichlet, save_folder=None, seed=0, split="train"):
    
    np.random.seed(seed)
    permut_1 = np.random.permutation(list(range(dataset_size)))
    permut_2 = np.random.permutation(list(range(dataset_size)))
    
    # With rough dirichlet per client, avoid having two distinct distributions in the federation
    total_size = dataset_size*2
    
    np.random.seed(seed)
    client_distrib = np.random.dirichlet(np.ones(2)*dirichlet, size=len(clients))
    print(client_distrib)
    
    max_samples = total_size/len(clients)
    
    nb_samples_per_origin_and_client_ndarray = (client_distrib*max_samples).round().astype(int)
    for row in range(len(clients)):
        while np.sum(nb_samples_per_origin_and_client_ndarray[row]) > max_samples:
            down_class = np.argmax(nb_samples_per_origin_and_client_ndarray[row])
            nb_samples_per_origin_and_client_ndarray[row, down_class] = nb_samples_per_origin_and_client_ndarray[row, down_class] - 1
         
    print(client_distrib.shape, client_distrib.sum(axis=0))
    print(nb_samples_per_origin_and_client_ndarray, nb_samples_per_origin_and_client_ndarray.sum(axis=0), nb_samples_per_origin_and_client_ndarray.sum(axis=1))
    
    origin_with_to_many_samples = np.argmax(nb_samples_per_origin_and_client_ndarray.sum(axis=0))
    
    while nb_samples_per_origin_and_client_ndarray.sum(axis=0)[0] != nb_samples_per_origin_and_client_ndarray.sum(axis=0)[1]:
        
        client_to_equilibrate = np.random.choice(np.arange(len(clients))[nb_samples_per_origin_and_client_ndarray[:,origin_with_to_many_samples] > 0])
        
        nb_samples_per_origin_and_client_ndarray[client_to_equilibrate, origin_with_to_many_samples] -= 1
        nb_samples_per_origin_and_client_ndarray[client_to_equilibrate, 1-origin_with_to_many_samples] += 1
        
    print(client_distrib.shape, client_distrib.sum(axis=0))
    print(nb_samples_per_origin_and_client_ndarray, nb_samples_per_origin_and_client_ndarray.sum(axis=0), nb_samples_per_origin_and_client_ndarray.sum(axis=1))

    current_1 = 0
    current_2 = 0
    final_assignement = {}
    for c in range(len(clients)):
        final_assignement[clients[c]] = {}
        if nb_samples_per_origin_and_client_ndarray[c, 0] > 0:
            final_assignement[clients[c]]["Clear"] = permut_1[current_1:current_1+nb_samples_per_origin_and_client_ndarray[c, 0]].tolist()
            current_1 += nb_samples_per_origin_and_client_ndarray[c, 0]
        if nb_samples_per_origin_and_client_ndarray[c, 1] > 0:
            final_assignement[clients[c]]["Inverted"] = permut_2[current_2:current_2+nb_samples_per_origin_and_client_ndarray[c, 1]].tolist()
            current_2 += nb_samples_per_origin_and_client_ndarray[c, 1]

    if save_folder is not None:
        with open(os.path.join(save_folder, f'{split}_partitioning_Inverted_Triple_MNIST.json'), 'w') as f:
            json.dump(final_assignement, f, indent=4)
            
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Clear"]) if "Clear" in final_assignement[c] else 0 for c in clients], label="Clear")
    plt.bar(list(range(len(clients))), [len(final_assignement[c]["Inverted"]) if "Inverted" in final_assignement[c] else 0 for c in clients], bottom=[len(final_assignement[c]["Clear"]) if "Clear" in final_assignement[c] else 0 for c in clients], label="Inverted")
    plt.title("Number of clear and inverted samples per client")
    plt.legend()
    plt.savefig(os.path.join(save_folder, f"Data_origin_distribution_per_client_{split}.svg"))

    return final_assignement
    

def assert_Inverted_Triple_MNIST_partitioning(final_assignement, clients):
    
    for c1 in clients:
        if "Clear" in final_assignement[c1]:
            for c2 in clients:
                if c1 != c2 and "Clear" in final_assignement[c2]:
                    print(f"{c1} vs {c2} for Clear", end="")
                    assert not bool(set(final_assignement[c1]["Clear"]) & set(final_assignement[c2]["Clear"]))
                    
    for c1 in clients:
        if "Inverted" in final_assignement[c1]:
            for c2 in clients:
                if c1 != c2 and "Inverted" in final_assignement[c2]:
                    print(f"{c1} vs {c2} for Inverted", end="")
                    assert not bool(set(final_assignement[c1]["Inverted"]) & set(final_assignement[c2]["Inverted"]))
                    
    print("\nClean partitioning")
