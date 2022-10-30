import torch
from torch_scatter import scatter_add
from visualize_mat import *

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    '''
    score_d: Shape [27332, 1]
        tensor([[0.],
                [0.],
                [0.],
                ...,
                [0.],
                [0.],
                [0.]])
    pos: Shape [1424, 3]
        tensor([[ -2.7890,  -1.3669,  -0.3078],
                [ -2.9441,   0.4065,  -0.6180],
                [ -1.3277,  -0.7638,  -1.4830],
                ...,
                [  3.4151,  16.8883, -12.5260],
                [  6.5336,  -8.5285,   5.9441],
                [  3.5570, -11.9453,  -6.6222]])
    edge_index: Shape [2, 27332]
        tensor([[   0,    0,    0,  ..., 1423, 1423, 1423],
                [   1,    2,    3,  ..., 1420, 1421, 1422]])
    edge_length: Shape [27332, 1]
        tensor([[ 1.8070],
                [ 1.9698],
                [ 0.7943],
                ...,
                [30.2063],
                [29.4321],
                [13.3583]])
    '''
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
    # save_mat_plt(dd_dr, 'visualize/dd_dr.png')
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    # save_heatmap_sns(dd_dr, 'visualize/score_pos.png')
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_angle(pos, angle_index):
    """
    Args:
        pos:  (N, 3)
        angle_index:  (3, A), left-center-right.
    """
    n1, ctr, n2 = angle_index   # (A, )
    v1 = pos[n1] - pos[ctr] # (A, 3)
    v2 = pos[n2] - pos[ctr]
    inner_prod = torch.sum(v1 * v2, dim=-1, keepdim=True)   # (A, 1)
    length_prod = torch.norm(v1, dim=-1, keepdim=True) * torch.norm(v2, dim=-1, keepdim=True)   # (A, 1)
    angle = torch.acos(inner_prod / length_prod)    # (A, 1)
    return angle


def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]   # (A, 3)
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1) # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)    # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) * torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)
    return dihedral


