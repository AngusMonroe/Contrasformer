"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gcn_net import GCNNet
from nets.contrasformer_net import ContrasformerNet


def GCN(net_params, trainset):
    return GCNNet(net_params)


def Contrasformer(net_params, trainset):
    return ContrasformerNet(net_params, trainset)


def gnn_model(MODEL_NAME, net_params, trainset):
    models = {
        'GCN': GCN,
        "Contrasformer": Contrasformer
    }
    model = models[MODEL_NAME](net_params, trainset)
    model.name = MODEL_NAME
        
    return model
