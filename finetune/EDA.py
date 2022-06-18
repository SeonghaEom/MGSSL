from loader import MoleculeDataset
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler

import os
import shutil

dataset_str = 'tox21'
dataset = MoleculeDataset("dataset/" + dataset_str, dataset=dataset_str)
smiles_list = pd.read_csv('dataset/' + dataset_str + '/processed/smiles.csv', header=None)[0].tolist()
train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)


# In[13]:


print(train_dataset[5])
print(train_dataset[5].x)
print(train_dataset[5].edge_index)
print(train_dataset[5].edge_attr)
print(train_dataset[5].y) ## if more than 1 length -> multilabel task


# In[68]:
def _get_cls_num(dataset):
  from collections import Counter
  y = []
  for i, row in enumerate(dataset):
    li = str(list(map(int, row.y)))
    y.append(li)

  cnt = Counter(y)
  return cnt

def get_cls_num(dataset, num_tasks):
  from collections import Counter
  y = []
  total_cnt = []
  for j in range(num_tasks):
    for i, row in enumerate(dataset):
      if row.y[j]!= 0 : y.append(int (row.y[j])) 
    cnt = Counter(y)
    total_cnt.append(cnt)
    y = []
  return total_cnt
  


def show_imbalance(dataset):
  ## get only y labels in NR-ER tasks
  y = []
  for i, row in enumerate(dataset):
    y.append(int(row.y[0]))

  # print(len(y), y)
  #
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  x_axis = ['1', '-1']
  y_axis = [y.count(1), y.count(-1)]
  ax.bar(x_axis, y_axis)
  plt.show()
  return 

# show_imbalance(train_dataset)
# show_imbalance(valid_dataset)
# show_imbalance(test_dataset)

def plot_result(log):
  with open(log, 'r') as f:
    contents = f.readlines()
    dataset = contents[0]
    results = contents[1:]
    y = [line.split()[-1] for i, line in results]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(y=y)
    plt.show()

## get weight for individual sample instance
def get_weighted_sampler(dataset):
  y = []
  for i, row in enumerate(dataset):
    y.append(int(row.y[0]))

  target = np.array(y)
  weight = dict()
  for t in np.unique(target):
      weight[t] = 1 / len(np.where(target == t)[0]) 
  # print(weight)
  samples_weight = np.array([weight[t] for t in target])
  samples_weight = torch.from_numpy(samples_weight)
  # print(samples_weight[:32])
  samples_weight = samples_weight.double()
  sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

  return sampler


# train_loader = DataLoader(train_dataset, batch_size=64, num_workers = 4, sampler=get_weighted_sampler(train_dataset))
# val_loader = DataLoader(valid_dataset, batch_size=32, num_workers = 4, sampler=get_weighted_sampler(valid_dataset))
# test_loader = DataLoader(test_dataset, batch_size=32, num_workers = 4,
# sampler=get_weighted_sampler(test_dataset))

# from collections import Counter
# for i, batch in enumerate(train_loader):
#   # print(len(batch.y))
#   resh = batch.y.reshape(-1, 12)[:,4].numpy()
#   counter = Counter(resh)
#   # print("batch {} 1/0/-1 {}/{}/{}".format(i, counter[1], counter[0], counter[-1]))

