import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from load_my_custom import *


def get_distribution(dataloader, count_dict):
    n = 0
    for batch, (X, y) in enumerate(dataloader):
        for label in y:
            count_dict[label.item()] +=1
    return count_dict 

def plot_dataset_distribution(dataloader, train, val, test, empty_dict):
    plt.figure()
    distribution = get_distribution(dataloader, empty_dict)
    x_axis = list(distribution.keys())
    y_ = list(distribution.values())
    print(x_axis)
    print(y_)

    if train:
        label = "train"
    if val:
        label = "validation"
    if test:
        label = "test"

    plt.bar(x_axis, y_, color = 'c', width = 0.4)
    plt.title("Distribution for the " + label + " dataset")
    plt.xlabel("Labels")
    plt.ylabel('Number of examples')
    plt.legend()
    plt.show()
    #plt.savefig(f"6_{type}_distribution_{label}_fps.png")
    

def main():
    keys = range(0, 9)
    class_distribution = {key: None for key in keys}
    for key in class_distribution:
        class_distribution[key] = 0
    print(class_distribution)

    keys = range(0, 9)
    empty_dict = {key: None for key in keys}
    for key in empty_dict:
        empty_dict[key] = 0
        
    data = get_data(1,transform)

    dataloader_train, dataloader_val, dataloader_test = create_dataloaders(data, 32)

    dict = get_distribution(dataloader_val, class_distribution)
    print(dict)
    plot_dataset_distribution(dataloader_test, 1, 0, 0, empty_dict)

