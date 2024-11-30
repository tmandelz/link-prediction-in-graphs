import argparse
from tqdm.auto import tqdm
from comet_ml import Experiment, Optimizer
import torch
import ast
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch.nn import Linear
from ogb.linkproppred import Evaluator
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from torch_sparse import SparseTensor
import random
import sys
sys.path.append('modelling/')
from evaluation.quali_evaluation_dl import Quali_Evaluator
from dataset_split.dataset_splitter import Dataset_Splitter


class NGNN_GCNConv(torch.nn.Module):
    def __init__(
        self, input_channels, hidden_channels, output_channels, num_layers
    ):
        super(NGNN_GCNConv, self).__init__()
        self.conv = GCNConv(input_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, output_channels)
        self.num_layers = num_layers

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, x, g, edge_weight=None):
        x = self.conv(x, g, edge_weight)
        if self.num_layers == 2:
            x = F.relu(x)
            x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(Linear(in_channels, hidden_channels), train_eps=True))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(Linear(hidden_channels, hidden_channels),train_eps=True))
        self.convs.append(GINConv(Linear(hidden_channels, out_channels),train_eps=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCN_NGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        ngnn_type,
    ):
        super(GCN_NGNN, self).__init__()
        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer
        if ngnn_type == "input":
            self.convs.append(
                NGNN_GCNConv(
                    in_channels,
                    hidden_channels,
                    hidden_channels,
                    num_nonl_layers,
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        elif ngnn_type == "hidden":
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    NGNN_GCNConv(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        num_nonl_layers,
                    )
                )

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, g):
        for conv in self.convs[:-1]:
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, g)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor: LinkPredictor, data, split_edge, optimizer, batch_size: int, experiment, epoch: int, step: int, seed: int, one_batch_training: bool = False):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0

    if one_batch_training:
        dataloader_train = [next(iter(DataLoader(range(source_edge.size(0)), batch_size,
                                                 shuffle=False)))]
        torch.manual_seed(seed)

    else:
        dataloader_train = DataLoader(range(source_edge.size(0)), batch_size,
                                      shuffle=True)

    for perm in tqdm(dataloader_train, desc='Training steps'):
        step += 1

        optimizer.zero_grad()

        # feed forward graph embedding
        h = model(data.x, data.adj_t)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        mask = dst_neg == src  # Find invalid negative edges
        while mask.any():
            dst_neg[mask] = torch.randint(0, data.num_nodes, (mask.sum().item(),), dtype=torch.long, device=h.device)
            mask = dst_neg == src  # Re-check
            
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples

        total_examples += num_examples
        experiment.log_metric("loss_step", loss.item(), epoch=epoch, step=step)

    return total_loss / total_examples, step


@torch.no_grad()
def test(model, predictor: LinkPredictor, data, split_edge, evaluator_mrr: Evaluator, evaluator_roc_auc: Evaluator, quali_evaluator: Quali_Evaluator, batch_size: int, experiment: Experiment, epoch: int, final_evaluation: bool = False):

    predictor.eval()
    h = model(data.x, data.adj_t)

    def get_test_split(split: str):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)
        return source, target, target_neg

    def test_split_quant(split: str, graph: dict, nx_graph: nx.classes.digraph.DiGraph, final_evaluation: bool):
        source, target, target_neg = get_test_split(split)
        # predicting the positive sampled links (ground-truth)
        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        # predicting the negative sampled links (generated-truth)
        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        # log metrics
        pos_loss = -torch.log(pos_pred + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_pred.view(-1) + 1e-15).mean()

        eval_dict_mrr = evaluator_mrr.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })
        eval_dict_roc_auc = evaluator_roc_auc.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred.view(-1),
        })

        def visualize_in_out_clustering_similarity(mrr_list, nx_graph, graph, train_edges, evaluation_nodes, validation=True):
            def plot_variable_against_mrr(variable, mrr, variable_name, validation):
                if validation:
                    dataset = "Validation"
                else:
                    dataset = "Test"
                jp = sns.jointplot(x=variable, y=mrr, kind='scatter', alpha=0.3, marginal_kws={
                                   'bins': 50, 'fill': True})
                jp.figure.suptitle(f'{dataset}: reciprocal rank vs {
                                   variable_name} \n n = {len(mrr)}', fontsize=20, y=1.05)
                jp.set_axis_labels(f'{variable_name}', 'reciprocal rank')
                plot_path = f"./temp/{dataset}_reciprocal_rank_vs_{variable_name}.png"
                jp.savefig(plot_path)
                plt.close(jp.figure)
                experiment.log_image(plot_path, name=f'{
                                     dataset}: reciprocal rank vs {variable_name}')

            def calculate_similarity_of_neighbours(train_edges, evaluation_nodes, mrr_list):
                adjacency_list = {}

                for source, target in zip(train_edges[0], train_edges[1]):
                    if source not in adjacency_list:
                        adjacency_list[source] = []
                    adjacency_list[source].append(target)
                sub_dict = {
                    key: adjacency_list[key] for key in evaluation_nodes if key in adjacency_list}
                mrr_vs_cosine_similarity = []
                for node in evaluation_nodes:
                    if node in adjacency_list:
                        mrr_vs_cosine_similarity.append([mrr_list[np.where(node == evaluation_nodes)][0],
                                                         cosine_similarity(np.array(graph.x[sub_dict[node]].cpu()), np.array([graph.x[node].cpu()])).mean()])

                return np.array(mrr_vs_cosine_similarity)[:, 1], np.array(mrr_vs_cosine_similarity)[:, 0]

            in_degrees = np.array(list(dict(nx_graph.in_degree()).values()))[
                evaluation_nodes]
            out_degrees = np.array(list(dict(nx_graph.out_degree()).values()))[
                evaluation_nodes]

            clustering_coefficients_for_plotting = [
                v for (k, v) in nx.clustering(nx_graph, evaluation_nodes).items()]

            similarity, mrr = calculate_similarity_of_neighbours(
                train_edges, evaluation_nodes, mrr_list)

            plot_variable_against_mrr(
                in_degrees, mrr_list, "Indegree", validation=validation)
            plot_variable_against_mrr(
                out_degrees, mrr_list, "out_degree", validation=validation)
            plot_variable_against_mrr(clustering_coefficients_for_plotting,
                                      mrr_list, "Clustering Coefficient", validation=validation)
            plot_variable_against_mrr(
                similarity, mrr, "Mean Cosine Similarity of Outdegree Neighbours", validation=validation)

        def mrr_distribution_connected_components(mrr_list, nx_graph, evaluation_nodes, validation=True):
            if validation:
                dataset = "Validation"
            else:
                dataset = "Test"
            nodes_in_strong_connected_components = set.union(
                *(set(s) for s in nx.strongly_connected_components(nx_graph) if len(s) != 1))
            nodes_in_strong = [
                node in nodes_in_strong_connected_components for node in evaluation_nodes]
            mrr_strong = mrr_list[nodes_in_strong]
            mrr_not_strong = mrr_list[~np.array(nodes_in_strong)]

            plt.figure(figsize=(10, 6))
            weights_strong = np.ones_like(mrr_strong) / len(mrr_strong)
            weights_not_strong = np.ones_like(
                mrr_not_strong) / len(mrr_not_strong)
            plt.hist(np.array(mrr_strong), bins=30, weights=weights_strong, alpha=0.5,
                     color="blue", label=f'Strong Connected Components n = {len(mrr_strong)} ')
            plt.hist(np.array(mrr_not_strong), bins=30, weights=weights_not_strong, alpha=0.5,
                     color="red", label=f'Not in Strong Connected Components n = {len(mrr_not_strong)}')
            plt.xlabel('reciprocal rank')
            plt.ylabel('Density')
            plt.title(f'{dataset}: reciprocal rank Distribution')
            plt.legend()
            plot_path = f"./temp/{dataset}_reciprocal_rank_Distribution.png"
            plt.savefig(plot_path)
            plt.close()
            experiment.log_image(plot_path, name=f'{
                                 dataset}: reciprocal rank Distribution')

        def calc_cosine_similarities(pos_pred, neg_pred, target_neg, source_unique, graph, target_val):
            graph.x = graph.x.cpu()
            target_neg = np.array(torch.tensor(target_neg).view(-1, 1000))
            selection_pos = (pos_pred > neg_pred.max(axis=1))
            proba_pos = pos_pred[selection_pos]
            positive_cosine_similarities = [cosine_similarity(graph.x[source].reshape(1, -1),
                                                              graph.x[target].reshape(1, -1))[0][0]
                                            for source, target
                                            in zip(source_unique[selection_pos], target_val[selection_pos])]

            probas_neg = neg_pred[~selection_pos].max(axis=1)
            neg_cos_sims = [cosine_similarity(graph.x[source].reshape(1, -1),
                                              graph.x[target].reshape(1, -1))[0][0]
                            for source, target
                            in zip(source_unique[~selection_pos],
                                   target_neg[~selection_pos, neg_pred[~selection_pos].argmax(axis=1)])]
            return proba_pos, positive_cosine_similarities, probas_neg, neg_cos_sims

        def visualize_cosine_similarities(pos_cos_sims, neg_cos_sims, validation=True):
            if validation:
                dataset = "Validation"
            else:
                dataset = "Test"

            weights_positive = np.ones_like(pos_cos_sims) / len(pos_cos_sims)
            weights_negative = np.ones_like(neg_cos_sims) / len(neg_cos_sims)

            overall_min = min(np.min(pos_cos_sims), np.min(neg_cos_sims))
            overall_max = max(np.max(pos_cos_sims), np.max(neg_cos_sims))

            plt.figure(figsize=(10, 6))
            plt.hist(pos_cos_sims, bins=50, range=(overall_min, overall_max), weights=weights_positive,
                     alpha=0.5, color="blue", label=f'Correct predicted n = {len(pos_cos_sims)}')
            plt.hist(neg_cos_sims, bins=50, range=(overall_min, overall_max), weights=weights_negative,
                     alpha=0.5, color="red", label=f'Wrong predicted n = {len(neg_cos_sims)}')
            plt.xlabel('Cosine Similarities')
            plt.ylabel('Density')
            plt.title(f'{dataset}:Distribution of Cosine Similarities')
            plt.legend()
            plot_path = f"./temp/{dataset}Distribution_of_Cosine_Similarities.png"
            plt.savefig(plot_path)
            plt.close()
            experiment.log_image(plot_path, name=f'{
                                dataset}: Distribution of Cosine Similarities')

        def visualize_proba_vs_cosine_similarity(probas_pos, pos_cos_sims, probas_neg, neg_cos_sims, validation=True):
            if validation:
                dataset = "Validation"
            else:
                dataset = "Test"
            plt.figure(figsize=(10, 6))
            plt.scatter(probas_neg, neg_cos_sims, color='red', alpha=0.3,
                        label=f'Wrong predicted n = {len(probas_neg)}', s=10)
            plt.scatter(probas_pos, pos_cos_sims, color='blue', alpha=0.3,
                        label=f'Correct predicted n = {len(probas_pos)}', s=10)
            plt.title(f'{dataset}: Word2vec Similarity of Top1 Predicted Node')
            plt.xlabel('Probability of Prediction')
            plt.ylabel('Cosine Similarities')
            plt.legend()
            plt.grid(True)
            plot_path = f"./temp/{dataset}Word2vec_Similarity_of_Top1_Predicted_Node.png"
            plt.savefig(plot_path)
            plt.close()
            experiment.log_image(plot_path, name=f'{
                                 dataset}: Word2vec Similarity of Top1 Predicted Node')

        def visualise_roc_curve(eval_dict_roc_auc, validation=True):
            if validation:
                dataset = "Validation"
            else:
                dataset = "Test"
            if 'fpr' in eval_dict_roc_auc.keys() and 'tpr' in eval_dict_roc_auc.keys():
                # Plot ROC curve
                plt.figure(figsize=(10, 6))
                auc = eval_dict_roc_auc['rocauc']
                plt.plot(eval_dict_roc_auc['fpr'], eval_dict_roc_auc['tpr'],
                         color='blue', label=f'ROC curve (AUC = {auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red',
                         linestyle='--')  # Diagonal line
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                plt.grid(True)
                plot_path = f"./temp/{dataset}roc_curve.png"
                plt.savefig(plot_path)
                plt.close()
                experiment.log_image(plot_path, name=f'{dataset}: roc_curve')

        if final_evaluation:
            if split == "valid":
                visualize_in_out_clustering_similarity(
                    eval_dict_mrr['mrr_list'], nx_graph, graph, train_edges, np.unique(np.array(source.cpu())), validation=True)
                mrr_distribution_connected_components(eval_dict_mrr['mrr_list'], nx_graph, np.unique(
                    np.array(source.cpu())), validation=True)
                p_pos, pos_cos_sim, p_neg, neg_cos_sim = calc_cosine_similarities(
                    np.array(pos_pred.clone().cpu()),
                    np.array(neg_pred.clone().cpu()),
                    np.array(target_neg.clone().cpu()),
                    np.unique(np.array(source.clone().cpu())),
                    graph, np.array(target.clone().cpu()))

                visualize_cosine_similarities(
                    pos_cos_sim, neg_cos_sim, validation=True)
                visualize_proba_vs_cosine_similarity(
                    p_pos, pos_cos_sim, p_neg, neg_cos_sim, validation=True)
                visualise_roc_curve(eval_dict_roc_auc, validation=True)
            if split == "test":
                visualize_in_out_clustering_similarity(eval_dict_mrr['mrr_list'], nx_graph, graph, train_edges, np.unique(
                    np.array(source.cpu())), validation=False)
                mrr_distribution_connected_components(eval_dict_mrr['mrr_list'], nx_graph, np.unique(
                    np.array(source.cpu())), validation=False)
                p_pos, pos_cos_sim, p_neg, neg_cos_sim = calc_cosine_similarities(
                    np.array(pos_pred.clone().cpu()),
                    np.array(neg_pred.clone().cpu()),
                    np.array(target_neg.clone().cpu()),
                    np.unique(np.array(source.clone().cpu())),
                    graph,
                    np.array(target.clone().cpu()))

                visualize_cosine_similarities(
                    pos_cos_sim, neg_cos_sim, validation=False)
                visualize_proba_vs_cosine_similarity(
                    p_pos, pos_cos_sim, p_neg, neg_cos_sim, validation=False)
                visualise_roc_curve(eval_dict_roc_auc, validation=False)

        return eval_dict_mrr['mrr_list'], eval_dict_roc_auc, (torch.unique(source), target, target_neg, pos_pred, neg_pred), pos_loss, neg_loss

    def test_split_quali(quali_evaluator: Quali_Evaluator, split: str, mrr_list: torch.tensor, graph, n_quali_edge_predictions: int, neg_pred, experiment, model, predictor, adj_t):
        """quali eval for a split, 

        Args:
            quali_evaluator (Quali_Evaluator): Evaluator Class for link prediction
            split (str): name of the split
            mrr_list (torch.tensor): list of mrr ranks
            n_quali_edge_predictions (int, optional): count of qualitative top and flop N observations. Defaults to 2.
        """

        topn_predicted_edges_indices = torch.sort(mrr_list, descending=True)[
            1][0:n_quali_edge_predictions]
        lown_predicted_edges_indices = torch.sort(mrr_list, descending=False)[
            1][0:n_quali_edge_predictions]

        source, target, target_neg = get_test_split(split)
        topn_edges = [(source[i], target[i])
                      for i in topn_predicted_edges_indices]
        flopn_edges = [(source[i], target[i])
                       for i in lown_predicted_edges_indices]

        topn = 0
        for (src, dst) in topn_edges:
            topn += 1
            if quali_evaluator.eval_n_hop_computational_graph > 0:
                quali_evaluator.calculate_and_visualize_computational_graph(f"{split} Top-{topn} Prediction, 1-Hop",
                                                                            int(src.cpu(
                                                                            )),
                                                                            np.unique(
                                                                                source.cpu()),
                                                                            neg_pred,
                                                                            target_neg.cpu(),
                                                                            int(dst.cpu(
                                                                            )),
                                                                            adj_t,
                                                                            graph,
                                                                            mrr_list,
                                                                            model,
                                                                            predictor,
                                                                            experiment,
                                                                            one_hop=True)
            if quali_evaluator.eval_n_hop_computational_graph > 1:
                quali_evaluator.calculate_and_visualize_computational_graph(f"{split} Top-{topn} Prediction, 2-Hop",
                                                                            int(src.cpu(
                                                                            )),
                                                                            np.unique(
                                                                                source.cpu()),
                                                                            neg_pred,
                                                                            target_neg.cpu(),
                                                                            int(dst.cpu(
                                                                            )),
                                                                            adj_t,
                                                                            graph,
                                                                            mrr_list,
                                                                            model,
                                                                            predictor,
                                                                            experiment,
                                                                            one_hop=False)

        flopn = 0
        for (src, dst) in flopn_edges:
            flopn += 1
            if quali_evaluator.eval_n_hop_computational_graph > 0:
                quali_evaluator.calculate_and_visualize_computational_graph(f"{split} Flop-{flopn} Prediction, 1-Hop",
                                                                            int(src.cpu(
                                                                            )),
                                                                            np.unique(
                                                                                source.cpu()),
                                                                            neg_pred,
                                                                            target_neg.cpu(),
                                                                            int(dst.cpu(
                                                                            )),
                                                                            adj_t,
                                                                            graph,
                                                                            mrr_list,
                                                                            model,
                                                                            predictor,
                                                                            experiment,
                                                                            one_hop=True)
            if quali_evaluator.eval_n_hop_computational_graph > 1:
                quali_evaluator.calculate_and_visualize_computational_graph(f"{split} Flop-{flopn} Prediction, 2-Hop",
                                                                            int(src.cpu(
                                                                            )),
                                                                            np.unique(
                                                                                source.cpu()),
                                                                            neg_pred,
                                                                            target_neg.cpu(),
                                                                            int(dst.cpu(
                                                                            )),
                                                                            adj_t,
                                                                            graph,
                                                                            mrr_list,
                                                                            model,
                                                                            predictor,
                                                                            experiment,
                                                                            one_hop=False)

    def log_arrays(path: str, variable: np.array, experiment=experiment):
        np.save("./temp/array_for_asset.npy", variable)
        experiment.log_asset("./temp/array_for_asset.npy", file_name=path)

    source = split_edge["train"]['source_node'].to(h.device)
    target = split_edge["train"]['target_node'].to(h.device)
    train_edges = np.vstack(
        (np.array([source.cpu()]), np.array([target.cpu()])))
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(data.x.shape[0]))
    nx_graph.add_edges_from(train_edges.T)

    train_mrr_list, train_eval_dict_roc_auc, _, train_pos_loss, train_neg_loss = test_split_quant(
        'eval_train', data, nx_graph, final_evaluation=final_evaluation)
    valid_mrr_list, valid_eval_dict_roc_auc, prediction_valid, valid_pos_loss, valid_neg_loss = test_split_quant(
        'valid', data, nx_graph, final_evaluation=final_evaluation)
    test_mrr_list, test_eval_dict_roc_auc, prediction_test, test_pos_loss, test_neg_loss = test_split_quant(
        'test', data, nx_graph, final_evaluation=final_evaluation)

    log_arrays(f"reciprocal_ranks/train/epoch_{epoch}", train_mrr_list)
    log_arrays(f"reciprocal_ranks/valid/epoch_{epoch}", valid_mrr_list)
    log_arrays(f"reciprocal_ranks/test/epoch_{epoch}", test_mrr_list)

    if final_evaluation:
        asset_names = ("source", "target", "target_neg",
                       "pos_preds", "neg_preds")

        test_split_quali(quali_evaluator, 'valid', valid_mrr_list, data,
                         15, prediction_valid[4], experiment, model, predictor, data.adj_t.cpu())
        for prediction_logging in range(len(prediction_valid)):
            log_arrays(f"prediction_logging/valid/{asset_names[prediction_logging]}.npy",
                       prediction_valid[prediction_logging].cpu())

        test_split_quali(quali_evaluator, 'test', test_mrr_list, data,
                         15, prediction_test[4], experiment, model, predictor, data.adj_t.cpu())
        for prediction_logging in range(len(prediction_test)):
            log_arrays(f"prediction_logging/test/{asset_names[prediction_logging]}.npy",
                       prediction_test[prediction_logging].cpu())

        if 'tpr' in valid_eval_dict_roc_auc.keys():
            log_arrays(f"prediction_logging/test/truepositiverates.npy",
                       valid_eval_dict_roc_auc['tpr'])
        if 'fpr' in valid_eval_dict_roc_auc.keys():
            log_arrays(f"prediction_logging/test/falsepositiverates.npy",
                       valid_eval_dict_roc_auc['fpr'])
        if 'thresholds' in valid_eval_dict_roc_auc.keys():
            log_arrays(f"prediction_logging/test/roc_thresholds.npy",
                       valid_eval_dict_roc_auc['thresholds'])

        if 'tpr' in test_eval_dict_roc_auc.keys():
            log_arrays(f"prediction_logging/test/truepositiverates.npy",
                       test_eval_dict_roc_auc['tpr'])
        if 'fpr' in test_eval_dict_roc_auc.keys():
            log_arrays(f"prediction_logging/test/falsepositiverates.npy",
                       test_eval_dict_roc_auc['fpr'])
        if 'thresholds' in test_eval_dict_roc_auc.keys():
            log_arrays(f"prediction_logging/test/roc_thresholds.npy",
                       test_eval_dict_roc_auc['thresholds'])

    return train_mrr_list.mean().item(), valid_mrr_list.mean().item(), test_mrr_list.mean().item(), train_eval_dict_roc_auc['rocauc'], valid_eval_dict_roc_auc['rocauc'], test_eval_dict_roc_auc['rocauc'], train_pos_loss, train_neg_loss, valid_pos_loss, valid_neg_loss, test_pos_loss, test_neg_loss


def init_gnn_model(args: dict, data: torch.tensor, device: str):
    if args.model_architecture == "GCN":
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.model_architecture == "GCN_NGNN":
        model = GCN_NGNN(data.num_features, args.hidden_channels,
                         args.hidden_channels, args.num_layers, args.dropout, args.ngnn_type).to(device)
    elif args.model_architecture == "SAGE":
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.model_architecture == "GIN":
        model = GIN(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)

    return model


def load_models(args, data, device):
    # init gnn node embedding
    model = init_gnn_model(args, data, device)
    # init model for link prediction
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    # load any previous checkpoints
    if isinstance(args.model_path, str):
        model.load_state_dict(torch.load(args.model_path))
    if isinstance(args.predictor_path, str):
        predictor.load_state_dict(torch.load(args.predictor_path))
    if args.freeze_model:
       for param in model.parameters():
           param.requires_grad = False

    return model, predictor


def init_optimizer(model, predictor, args):
    # init optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)

    return optimizer


def log_hyper_params(experiment: Experiment, args, run_index: int, device):
    """adds hyperparams to a dict and logs them in comet ml

    Args:
        experiment (Experiment): comet ml experiment
        args (_type_): args from parseargs
        run_index (int): index of the run executing
        device (_type_): cuda deivce index
    """
    hyper_params = {
        "device": device,
        "learning_rate": args.lr,
        "epoch": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_n_hop_computational_graph": args.eval_n_hop_computational_graph,
        "n_runs": args.runs,
        "dropout": args.dropout,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "one_batch_training": args.one_batch_training,
        "random_seed": args.random_seed,
        "model_architecture": args.model_architecture,
        "dataset": args.dataset,
        "save_model": args.save_model,
        "model_path":  args.model_path,
        "predictor_path": args.predictor_path,
        "freeze_model": args.freeze_model
    }
    hyper_params["run"] = run_index
    experiment.log_parameters(hyper_params)


def unzip_test_results(result, loss=None):
    train_mrr, valid_mrr, test_mrr, train_rocauc, valid_rocauc, test_rocauc, train_pos_loss, train_neg_loss, valid_pos_loss, valid_neg_loss, test_pos_loss, test_neg_loss = result
    metrics = {
        "train_mrr": train_mrr,
        "valid_mrr": valid_mrr,
        "test_mrr": test_mrr,
        "train_rocauc": train_rocauc,
        "valid_rocauc": valid_rocauc,
        "test_rocauc": test_rocauc,
        "train_pos_loss": train_pos_loss,
        "train_neg_loss": train_neg_loss,
        "train_loss_for_testing": train_pos_loss + train_neg_loss,
        "valid_pos_loss": valid_pos_loss,
        "valid_neg_loss": valid_neg_loss,
        "valid_loss": valid_pos_loss + valid_neg_loss,
        "test_pos_loss": test_pos_loss,
        "test_neg_loss": test_neg_loss,
        "test_loss": test_pos_loss + test_neg_loss
    }
    if loss is not None:
        metrics["loss_epoch"] = loss
    return metrics


def dict_type(arg):
    try:
        # Use ast.literal_eval to safely evaluate the string as a dictionary
        return ast.literal_eval(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid dictionary: {}".format(arg))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='GCN Trainer')
    parser.add_argument('--device', type=int, default=0)  # default 0
    parser.add_argument('--num_layers', type=int, default=2)  # default 3
    parser.add_argument('--hidden_channels', type=int,
                        default=512)  # default 256
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int,
                        default=38349)  # default 64*1024
    parser.add_argument('--eval_batch_size', type=int,
                        default=100000)  # default 100000
    parser.add_argument('--eval_n_hop_computational_graph',
                        type=int, default=2)  # default 2
    parser.add_argument('--lr', type=float, default=0.00004)  # default 0.0005
    parser.add_argument('--epochs', type=int, default=2000)  # default 50
    parser.add_argument('--runs', type=int, default=1)  # default 10
    parser.add_argument('--project_name', type=str,
                        default="link-prediction-development")
    parser.add_argument('--run_name', type=str, default="test_ngnn")
    parser.add_argument('--dataset', type=str,
                        default="ogbn-arxiv")  # defaGult ogbn-arxiv
    parser.add_argument('--model_architecture', type=str,
                        default="GIN", choices=['GCN', 'GCN_NGNN', 'SAGE', 'GIN'])
    parser.add_argument('--ngnn_type', type=str,
                        default="input", choices=['input', 'hidden'])
    parser.add_argument('--one_batch_training', type=str2bool,
                        default=False, choices=[False, True])  # default False
    parser.add_argument('--random_seed', type=int,
                        default=12345)  # default 12345
    parser.add_argument('--parameter_tuning_algo', type=str, default="random",
                        choices=["random", "grid", "bayes"])  # default random
    parser.add_argument('--parameter_tuning_param_grid',
                        type=dict_type, default=None)  # default None
    parser.add_argument('--save_model', type=str2bool,
                        default=False, choices=[False, True])  # default False
    parser.add_argument('--epoch_checkpoints', type=int,
                        default=100)  # default False
    parser.add_argument('--model_path', type=str, default=None)  # default None
    parser.add_argument('--predictor_path', type=str,
                        default=None)  # default None
    parser.add_argument('--freeze_model', type=str2bool,
                        default=False, choices=[False, True])  # default False
    parser.add_argument('--neg_sample_size', type=int,
                        default=1000)  # default 1000

    args = parser.parse_args()

    # set seeds for reproducibility, only for numpy used in dataset split
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # set device
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if isinstance(args.parameter_tuning_param_grid, type(dict())):
        parameter_tuning = True
        # defining the configuration dictionary for comet ml parameter tuning
        config_dict = {"algorithm": args.parameter_tuning_algo,
                       "spec": {
                           "metric": "valid_mrr",
                           "objective": "maximize",
                           "maxCombo": 100,
                           "gridSize": 10,
                       },
                       "parameters": args.parameter_tuning_param_grid,
                       "name": "Link-Prediction-Optimizer",
                       "trials": 1,
                       }

        # initializing the comet ml optimizer
        opt = Optimizer(config=config_dict,
                        api_key="fMjtHh9OnnEygtraNMjP7Wpig",
                        project_name=args.project_name,
                        workspace="swiggy123")

        # iterator consists of experiments
        run_iterator = opt.get_experiments()
    else:
        # iterator consists of run 0 to x
        parameter_tuning = False
        run_iterator = range(args.runs)

    # load dataset
    ds_split = Dataset_Splitter(args.neg_sample_size)
    dataset = ds_split.load_dataset(args.dataset)
    split_edge, edge_index = ds_split.get_edges_split(dataset)

    data = dataset[0]

    # added
    row, col = edge_index
    data.adj_t = SparseTensor(
        row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes))
    data = data.to(device)

    # set eval size for trainset by taking the size of the validation
    eval_size_trainset = split_edge['valid']['source_node'].shape[0]
    # We randomly pick training samples
    idx = torch.randperm(split_edge['train']['source_node'].numel())[
        :eval_size_trainset]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    if args.model_architecture == "GCN" or args.model_architecture == "GCN_NGNN":
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
    else:
        pass

    # defines mrr as the eval metric
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    evaluator_roc_auc = Evaluator(name='ogbl-vessel')
    quali_evaluator = Quali_Evaluator(
        data.x, edge_index, args.eval_n_hop_computational_graph)

    for run_index, run in tqdm(enumerate(run_iterator), desc="Training Runs"):
        if not parameter_tuning:
            # init a new experiment per run
            experiment = Experiment(
                api_key="fMjtHh9OnnEygtraNMjP7Wpig",
                project_name=args.project_name,
                workspace="swiggy123"
            )
            experiment.set_name(f"{args.run_name}")
        else:
            experiment = run
            for k, v in experiment.params.items():
                # overwrite args with parameter tuning args
                setattr(args, k, v)
            experiment.set_name(f"{args.run_name}-Parametertuning-{run_index}")

        # log hyper params
        log_hyper_params(experiment, args, run_index, device)

        # init model and optimizer or load them from previous checkpoint
        model, predictor = load_models(args, data, device)
        optimizer = init_optimizer(model, predictor, args)

        step = 0
        print(f"Start Training: \nArgs:{args}")

        for epoch in tqdm(range(1, 1 + args.epochs), desc='Training Epochs'):
            # log before model started training, used for transfer learning metrics (jumpstart)
            result = test(model, predictor, data, split_edge, evaluator_mrr, evaluator_roc_auc, quali_evaluator,
                          args.eval_batch_size, experiment, epoch-1)
            data.x, model, predictor = [variable.to(
                device) for variable in (data.x, model, predictor)]

            metrics = unzip_test_results(result)
            experiment.log_metrics(metrics, epoch=epoch-1)

            loss, step = train(model, predictor, data, split_edge, optimizer,
                               args.batch_size, experiment, epoch, step, args.random_seed, args.one_batch_training)

            print(
                f'Run: {run_index + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            experiment.log_metrics({"loss_epoch": loss}, epoch=epoch)
            experiment.log_epoch_end(epoch, step)
            if args.save_model:
                if (epoch % args.epoch_checkpoints) == 0:
                    torch.save(model.state_dict(),
                               f"./temp/model_epoch_{epoch}.pth")
                    torch.save(predictor.state_dict(),
                               f"./temp/predictor_epoch_{epoch}.pth")
                    experiment.log_model(
                        f"./model_checkpoints/model_epoch_{epoch}", f"./temp/model_epoch_{epoch}.pth")
                    experiment.log_model(
                        f"./predictor_checkpoints/predictor_epoch_{epoch}", f"./temp/predictor_epoch_{epoch}.pth")

        # save model checkpoint if needed
        if args.save_model:
            torch.save(model.state_dict(), "./temp/model.pth")
            torch.save(predictor.state_dict(), "./temp/predictor.pth")
            experiment.log_model("model", "./temp/model.pth")
            experiment.log_model("predictor", "./temp/predictor.pth")
        # log qualitative after last epoch
        result = test(model, predictor, data, split_edge, evaluator_mrr, evaluator_roc_auc, quali_evaluator,
                      args.eval_batch_size, experiment, epoch, final_evaluation=True)
        data.x, model, predictor = [variable.to(
            device) for variable in (data.x, model, predictor)]

        metrics = unzip_test_results(result, loss)
        experiment.log_metrics(metrics, epoch=epoch)
        experiment.log_epoch_end(epoch, step)

        experiment.end()

        # remove memory from GPU
        del model, predictor, optimizer


if __name__ == "__main__":
    main()
