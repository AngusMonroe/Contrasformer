import torch


def get_summary_feat(G_dataset, Labels, merge_classes=False):
    num_G = len(G_dataset)
    Labels = Labels.tolist()
    nodes_dict = {}
    final_nodes_dict = {}
    for i in range(num_G):
        if Labels[i] not in nodes_dict.keys():
            nodes_dict[Labels[i]] = []
        nodes_dict[Labels[i]].append(G_dataset[i].ndata['feat'].tolist())

    if merge_classes:
        hc_idx, ad_idx = sorted(nodes_dict.keys())[0], sorted(nodes_dict.keys())[-1]
        final_nodes_dict[0] = torch.tensor(nodes_dict[hc_idx])
        final_nodes_dict[1] = torch.tensor(nodes_dict[ad_idx])
    else:
        for i in nodes_dict.keys():
            final_nodes_dict[i] = torch.tensor(nodes_dict[i])
    return final_nodes_dict


def cal_summary_graph(G_dataset, Labels):
    idx_A = (Labels != 0).nonzero()
    idx_B = (Labels == 0).nonzero()

    num_G = len(G_dataset)
    num_A = len(idx_A)
    num_B = len(idx_B)

    adj_A = None
    adj_B = None
    for i in range(num_G):
        if i in idx_A:
            if adj_A is None:
                adj_A = G_dataset[i].adj()
            else:
                adj_A = adj_A + G_dataset[i].adj()
        else:
            if adj_B is None:
                adj_B = G_dataset[i].adj()
            else:
                adj_B = adj_B + G_dataset[i].adj()
    adj_A = (adj_A / num_A).to_dense()
    adj_B = (adj_B / num_B).to_dense()

    print('edge# of summary graph A: {}'.format(len(adj_A.reshape(-1).nonzero())))
    print('edge# of summary graph B: {}'.format(len(adj_B.reshape(-1).nonzero())))

    adj_A = adj_A.numpy()
    adj_B = adj_B.numpy()

    return adj_A, adj_B
