# Motif-based Graph Self-Supervised Learning for Molecular Property Prediction
Reproduction of official Pytorch implementation of NeurIPS'21 paper "Motif-based Graph Self-Supervised Learning for Molecular Property Prediction"
(https://arxiv.org/abs/2110.00987). You have to be careful with rdkit version in order to successfully reproduce all motif vocabulary
## Requirements
```
pytorch                   1.8.1             
torch-geometric           1.7.0
rdkit                     2020.09.1
tqdm                      4.31.1
tensorboardx              1.6
```
To install RDKit, please follow the instructions here http://www.rdkit.org/docs/Install.html

* `motif_based_pretrain/` contains codes for motif-based graph self-supervised pretraining.
* `finetune/` contains codes for finetuning on MoleculeNet benchmarks for evaluation.
## Training
You can pretrain the model by
```
cd motif_based_pretrain
python pretrain_motif.py
```

## Evaluation
You can evaluate the pretrained model by finetuning on downstream tasks
```
cd finetune
python finetune.py
```
