# SMIA

This is the implementation for our paper "Subgraph Structure Membership Inference Attacks against Graph Neural Networks", which has been accepted by PoPETs 2024.

## GNNs (target model)

The original implemenations of GNN models we used in the paper can be found here:

- GCN: https://github.com/tkipf/pygcn

- the implementation of both GraphSAGE and GAT from DGL package: https://github.com/dmlc/dgl

## Requirements

To run the code of GNNs, please use the environments and required packages from the links above:

 - for GCN, use PyTorch 0.4 or 0.5, Python 2.7 or above

 - for GraphSAGE and GAT, import the package of DGL

## Step1: Run three GNNs on three datasets to get the embeddings, posteriors

    python train-gcn.py, python train_graphsage.py, python train_gat.py 

## Step2: Run the attack models

    train-popets-3smia.py

    train-popets-4smia.py
 
## Step3: Evaluate the defense mechanisms

    train-popets-3smia-defense.py

    train-popets-4smia-defense.py




