import argparse

from loader import MoleculeDataset
from losses import LDAMLoss
from EDA import get_weighted_sampler, get_cls_num
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil
import logging
from pathlib import Path
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
Path("log/").mkdir(parents=True, exist_ok=True)



from tensorboardX import SummaryWriter





def train(args, model, device, loader, optimizer, criterion):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        # print(pred.shape, y.shape)
        # print((y+1)/2)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, criterion):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    prc_list = []
    recall_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            prc_list.append(average_precision_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            recall_list.append(recall_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), sum(prc_list)/len(prc_list), sum(recall_list)/len(recall_list) #y_true.shape[1], 



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--weight_sample', action='store_true')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '../motif_based_pretrain/saved_model/motif_pretrain.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()
    fh = logging.FileHandler('log/{}_{}.log'.format(args.dataset, args.weight_sample))
    logger.addHandler(fh)
    logger.info(args.dataset)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    # print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])
    
    

    if args.weight_sample:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_workers, sampler=get_weighted_sampler(train_dataset))
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers = args.num_workers, sampler=get_weighted_sampler(valid_dataset))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers = args.num_workers, sampler=get_weighted_sampler(test_dataset))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_workers )
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers = args.num_workers )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers = args.num_workers )

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # print(optimizer)

    cls_nums = get_cls_num(train_dataset, num_tasks=num_tasks) #class key: counts - for each task
    cls_num_list = []
    for each in cls_nums:
        assert set(each.keys()) == set([-1, 1]), each.keys()
        cls_num_list.append(float(each[-1]/each[1]))
    criterion = LDAMLoss(np.array(cls_nums))
    print("cls num list ", cls_num_list)
    # pos_weight = torch.FloatTensor(cls_num_list).to(device)
    # criterion = nn.BCEWithLogitsLoss(reduction = "none", pos_weight=pos_weight)
        

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer, criterion)

        print("====Evaluation")
        if args.eval_train:
            train_acc, train_prc, train_rec = eval(args, model, device, train_loader, criterion)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc, val_prc, val_rec = eval(args, model, device, val_loader, criterion)
        test_acc, test_prc, test_rec = eval(args, model, device, test_loader, criterion)

        print("ACC train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        print("PRC train: %f val: %f test: %f" %(train_prc, val_prc, test_prc))
        print("REC train: %f val: %f test: %f" %(train_rec, val_rec, test_rec))
        logger.info("ACC train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        logger.info("PRC train: %f val: %f test: %f" %(train_prc, val_prc, test_prc))
        logger.info("REC train: %f val: %f test: %f" %(train_rec, val_rec, test_rec))

if __name__ == "__main__":
    main()
