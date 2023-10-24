import h5py
import numpy as np


def readH5File(path):
    file = h5py.File(path, 'r')
    feature = np.empty((0, 1024))
    label_list = []
    for dataset_name in file:
        label_list.append(dataset_name)
        dataset = file[dataset_name]
        feature = np.concatenate((feature, dataset[:].reshape(1, -1)))
    file.close()
    return feature, label_list