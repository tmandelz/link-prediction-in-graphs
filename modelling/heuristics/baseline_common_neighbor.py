import argparse
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from torch_geometric.data import DataLoader
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)

from ogb.linkproppred import Evaluator
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from tqdm import tqdm
import random
import sys

# adapted from https://github.com/facebookresearch/SEAL_OGB

sys.path.append('modelling/')
from dataset_split.dataset_splitter import Dataset_Splitter

def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long,)
def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()

        if 'edge_neg' in split_edge[split]:
            # use presampled negative training edges for ogbl-vessel
            neg_edge = split_edge[split]['edge_neg'].t()

        else:
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))

        # subsample for pos_edge
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1000],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge

def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results

def evaluate_mrr(pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator):

    neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    return train_mrr, valid_mrr, test_mrr


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results

def evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,evaluator):
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
        })[f'rocauc']

    test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']

    results = {}
    results['rocauc'] = (valid_rocauc, test_rocauc)
    return results



def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def main():
    parser = argparse.ArgumentParser(description='Baselines Trainer')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--project_name', type=str, default="link_prediction-baselines")
    parser.add_argument('--run_name', type=str, default="common_neighbor_arxiv")
    parser.add_argument('--eval_metric', type=str, default="mrr")
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv")
    parser.add_argument('--model_architecture', type=str, default="CN")
    parser.add_argument('--random_seed', type=int, default=12345)

    args = parser.parse_args()
    print(args)
    
    # set device
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # set up experiment
    experiment = Experiment(
        api_key="fMjtHh9OnnEygtraNMjP7Wpig",
        project_name=args.project_name,
        workspace="swiggy123"
    )
    
    # load dataset
    ds_split = Dataset_Splitter()
    dataset = ds_split.load_dataset(args.dataset)
    split_edge, edge_index = ds_split.get_edges_split(dataset)

    data = dataset[0]

    # added
    data = data.to(device)
    
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
    
    hyper_params = {
        "device": device,
        "run_name": args.run_name
    }
        
    # set random seed and change it per run
    np.random.seed(args.random_seed)
    
    experiment.log_parameters(hyper_params)
    experiment.set_name(args.run_name)
    # create a sparse adjacency matrix
    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])),
                       shape=(num_nodes, num_nodes))
    pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                       data.edge_index,
                                                       data.num_nodes)
    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge,
                                                   data.edge_index,
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge,
                                                     data.edge_index,
                                                     data.num_nodes)

    pos_train_pred, pos_train_edge = eval("CN")(A, pos_train_edge)
    neg_train_pred, neg_train_edge = eval("CN")(A, neg_train_edge)
    pos_val_pred, pos_val_edge = eval("CN")(A, pos_val_edge)
    neg_val_pred, neg_val_edge = eval("CN")(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval("CN")(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval("CN")(A, neg_test_edge)
    
    # create evaluator
    evaluator = Evaluator(name='ogbl-citation2')
    if args.eval_metric == 'hits':
        raise NotImplementedError()
        # results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator)
    elif args.eval_metric == 'mrr':
        # evaluate mrr
        train_mrr, valid_mrr, test_mrr = evaluate_mrr(pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, evaluator)
        
        # log metrics
        metrics = {
        "train_mrr": train_mrr,
        "valid_mrr": valid_mrr,
        "test_mrr": test_mrr}
        experiment.log_metrics(metrics)
    elif args.eval_metric == 'rocauc':
        raise NotImplementedError()
        # results = evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,evaluator)
    elif args.eval_metric == 'auc':
        raise NotImplementedError()
        # val_pred = torch.cat([pos_val_pred, neg_val_pred])
        # val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
        #                     torch.zeros(neg_val_pred.size(0), dtype=int)])
        # test_pred = torch.cat([pos_test_pred, neg_test_pred])
        # test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
        #                     torch.zeros(neg_test_pred.size(0), dtype=int)])
        # results = evaluate_auc(val_pred, val_true, test_pred, test_true)
    
    experiment.end()
    



if __name__ == "__main__":
    main()