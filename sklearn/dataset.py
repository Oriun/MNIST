from genericpath import exists
import gzip
from operator import index
import numpy as np
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import os

class Dataset():
    labels = []
    images = []
    image_size = 28
    distribution = {}
    do_print= True
    def __init__(self, training = False, print=True):
        self.do_print = print
        self.training = training
        self.print('Creating Dataset')
        self.labels_path = "../datasets/"
        self.image_path = "../datasets/"
        if training:
            self.image_path+= "train"
            self.labels_path+= "train"
        else:
            self.image_path += "t10k"
            self.labels_path += "t10k"
            
        self.image_path += "-images-idx3-ubyte.gz"
        self.labels_path += "-labels-idx1-ubyte.gz"
        self.pull_data()
        
    def print(self, *args):
        if self.do_print:
            print(*args)
        
    def pull_data(self):
        self.print('Pulling Data')
        
        ##### Pull images
        self.print('Image Buffer Read')
        raw_images = gzip.open(self.image_path,"r")
        raw_images.read(16)
        images_buff = np.frombuffer(raw_images.read(), dtype=np.uint8)
        self.print('Shaping')
        self.images = images_buff.reshape(floor(len(images_buff) / self.image_size**2), self.image_size, self.image_size, 1)
        self.print('Ok')
        
        ##### Pull labels
        self.print('Label Buffer Read')
        raw_labels = gzip.open(self.labels_path,'r')
        raw_labels.read(8)
        self.labels = np.frombuffer(raw_labels.read(), dtype=np.uint8)
        self.print('Ok')
        
        unique, counts = np.unique(self.labels, return_counts=True)
        for i in range(len(unique)):
            self.distribution[unique[i]] = counts[i]

    def reshape(self):
        self.print('Reshaping images...')
        return list(map(lambda image: np.reshape(image,(1,784)).tolist()[0], self.images))
            
    def display_distribution(self):
        self.print('Display distribution')
        values = list(self.distribution.values())
        keys = list(self.distribution.keys())
        bar = plt.bar([i for i, _ in enumerate(keys)], values) 
        plt.xticks(keys)
        for rect, label in zip(bar.patches, list(self.distribution.values())):
            height = rect.get_height()
            plt.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
            )
        plt.title("Distribution des chiffres dans le dataset") 
        plt.show()
        
if __name__ == "__main__":
    training = Dataset(True)
    print(training.distribution)
    testing = Dataset(False)
    print(testing.distribution)
