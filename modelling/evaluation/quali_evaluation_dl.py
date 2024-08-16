import matplotlib.pyplot as plt
import torch
import networkx as nx
import numpy as np
from captum.attr import IntegratedGradients
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import itertools
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from torch_sparse import SparseTensor


class Quali_Evaluator():
    def __init__(self, node_features, edge_index, eval_n_hop_computational_graph):
        self.node_features = node_features.cpu()
        self.edge_index = edge_index.cpu()
        self.eval_n_hop_computational_graph = eval_n_hop_computational_graph

        self.adjacency_matrix = self.create_adjacency_matrix(self.edge_index)
        print(f"Quali_Evaluator is initialized")
        pass

    def get_embedding_edge(self, node_source_index: int, node_target_index: int):
        embedding_source_node = self.embedding[node_source_index]
        embedding_target_node = self.embedding[node_target_index]
        return embedding_source_node, embedding_target_node

    def create_adjacency_matrix(self, edge_index):
        num_nodes = int(torch.max(edge_index).item()) + 1
        # Create adjacency matrix from edge index
        values = torch.ones(edge_index.shape[1])
        adj_matrix = torch.sparse_coo_tensor(
            edge_index, values, (num_nodes, num_nodes))
        sparse_adj_matrix = adj_matrix.coalesce()

        return sparse_adj_matrix

    @staticmethod
    def select_source_target_nodes(source_node, source_unique, negativ_pred, target_negativ):
        position_selected = np.where(source_node == source_unique)[0][0]
        target_negativ_selected = target_negativ[position_selected][negativ_pred[position_selected].argmax(
        )]
        return target_negativ_selected, position_selected

    @staticmethod
    def calc_all_nodes(adjacency_list, source_node, target_node, hopp_2: bool, train_edges):
        nodes_selected = [target_node, source_node] + list(
            train_edges[1][train_edges[0] == target_node]) + list(train_edges[1][train_edges[0] == source_node])
        if hopp_2:
            nodes_selected = np.unique(np.array(list(itertools.chain.from_iterable(
                [adjacency_list[neighbour] for neighbour in nodes_selected if neighbour in adjacency_list])) + nodes_selected))
        return np.unique(nodes_selected)

    @staticmethod
    def calc_low_dim_features(graph: dict, nodes_selected: list):
        node_features = graph.x[nodes_selected]
        high_dim_cosine_distances = pairwise_distances(
            node_features, metric='cosine')
        high_dim_cosine_distances = (
            high_dim_cosine_distances + high_dim_cosine_distances.T) / 2  # for numerical stability
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        low_dimensional_data = mds.fit_transform(high_dim_cosine_distances)
        low_dim_cosine_distances = pairwise_distances(
            low_dimensional_data, metric='euclidean')
        nodes_distance = np.abs(
            low_dim_cosine_distances.ravel() - high_dim_cosine_distances.ravel())
        return low_dimensional_data, nodes_distance.mean(), nodes_distance.max()

    @staticmethod
    def select_edges_from_nodes(nodes_selected, train_edges):
        selected_edges = np.array([edge for edge in train_edges.T if (
            edge[0] in nodes_selected) and (edge[1] in nodes_selected)])
        node_mapping = {old_idx: new_idx for new_idx,
                        old_idx in enumerate(nodes_selected)}
        remapped_edges = np.vectorize(node_mapping.get)(selected_edges)
        remapped_edges_with_diag = np.hstack((remapped_edges.T, np.array(
            [list(range(len(nodes_selected))), list(range(len(nodes_selected)))])))
        return selected_edges, node_mapping, remapped_edges_with_diag

    @staticmethod
    def get_neighbours(adj_t,source,target):
        col = adj_t.storage.col()
        row = adj_t.storage.row()


        def find_neighbors(row,col, index):
            neighbors = col[row == index]
            return neighbors

        # Find neighbors of start and target indices
        start_neighbors = find_neighbors(row,col, source)
        target_neighbors = find_neighbors(row,col, target)

        # Find neighbors of neighbors
        start_neighbors_of_neighbors = torch.cat([find_neighbors(row,col, neighbor) for neighbor in start_neighbors])
        target_neighbors_of_neighbors = torch.cat([find_neighbors(row,col, neighbor) for neighbor in target_neighbors])

        # Combine all unique indices to form the new adjacency matrix
        all_indices = torch.unique(torch.cat([torch.tensor([source]), torch.tensor([target]), start_neighbors, target_neighbors, start_neighbors_of_neighbors, target_neighbors_of_neighbors]))
        indices_one_neighbour = torch.unique(torch.cat([torch.tensor([source]), torch.tensor([target]), start_neighbors, target_neighbors]))
        return all_indices,indices_one_neighbour
    
    @staticmethod
    def create_sparse_tensor(indices,adj_t):
        mask_row = torch.isin(adj_t.storage.row(), indices)
        mask_col = torch.isin(adj_t.storage.col(), indices)
        mask = mask_row & mask_col

        # Filter the rows, columns, and values
        new_row = adj_t.storage.row()[mask]
        new_col = adj_t.storage.col()[mask]
        new_val = adj_t.storage.value()[mask]

        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(np.array(indices))}
        row = [node_mapping[row_value] for row_value in np.array(new_row)]
        col = [node_mapping[col_value] for col_value in np.array(new_col)]
        sparse_tensor_neighbour = SparseTensor(row=torch.tensor(row), col=torch.tensor(col),value=new_val, sparse_sizes=(len(indices), len(indices)))
        return sparse_tensor_neighbour,node_mapping


    @staticmethod
    def calculate_ig_link_predictions(node_mapping,source_node,target,nodes_selected,model,predictor,graph,edges_sparse):
        def model_wrapper(x, edge_index):
            embeddings = model.cpu()(x, edge_index)
            return predictor.cpu()(embeddings[node_mapping[source_node]], embeddings[node_mapping[int(target)]])

        def ig_link_prediction(model, predictor, x_tensor, edge_index):
            model.eval()
            predictor.eval() 
            device = next(model.parameters())

            x_tensor = x_tensor.float()
            edge_index = edge_index

            ig = IntegratedGradients(lambda x: model_wrapper(x, edge_index))

            baseline_x = torch.zeros_like(x_tensor)

            attribute = ig.attribute(inputs=(x_tensor,),
                                    baselines=(baseline_x,),
                                    target=None,
                                    internal_batch_size=len(node_mapping.keys()),
                                    return_convergence_delta=True,
                                    n_steps=100)

            return attribute

        x_tensor = torch.tensor(graph.x[nodes_selected], dtype=torch.float)
        edge_index = edges_sparse

        attribute = ig_link_prediction(model, predictor, x_tensor, edge_index)
        node_importances = np.array(attribute[0][0].sum(axis=1))
        return node_importances/ max(node_importances)

    @staticmethod
    def visualize_computational_graph(ax, low_dimensional_data, nodes_selected, edges, source_node, target, average_distance, maximal_distance, mrr_list, position_selected, feature_values,title,node_mapping):

        G = nx.DiGraph()
        for i, pos in enumerate(low_dimensional_data):
            G.add_node(i, pos=pos)
        G.add_edges_from(edges)
        positions = {node: data['pos'] for node, data in G.nodes(data=True)}
        # Normalize feature values for color mapping
        norm = Normalize(vmin=min(feature_values), vmax=max(feature_values))
        cmap = plt.get_cmap('viridis')

        # Draw all nodes with colors based on feature values
        colors = [cmap(norm(value)) for value in feature_values]
        nx.draw(G, pos=positions, node_size=200, node_color=colors, with_labels=False, arrows=True, arrowstyle='-|>', arrowsize=12, ax=ax)

        # Highlight the special edge
        special_edge = (node_mapping[source_node], node_mapping[int(target)])
        nx.draw_networkx_edges(G, pos=positions, edgelist=[special_edge], width=2, edge_color='red', style='dashed', arrows=True, arrowstyle='-|>', arrowsize=12, ax=ax)

        # Create colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Node Importance')


        ax.set_title(title, y=1.15)
        ax.text(0.5, 1.08, 
                f"Rang: {round(1/np.array(mrr_list)[position_selected])} \n Avg. Distance Mapping Error: {average_distance:.4f} \n Max. Distance Mapping Error: {maximal_distance:.4f} \n Source Node Index: {source_node} \n Target Node Index: {int(target)}",
                fontsize=14, ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.grid(True)


    def calculate_and_visualize_computational_graph(self, prefix_title, source_node, source_unique, neg_pred, target_neg, target_positiv_selected, adj_t, graph, mrr_list, model, predictor, experiment, one_hop):
        target_negativ_selected, position_selected = Quali_Evaluator.select_source_target_nodes(
            source_node, source_unique, neg_pred, target_neg)
        nodes_selected_negativ,indices_one_neighbour_negativ = Quali_Evaluator.get_neighbours(adj_t,source_node,target_negativ_selected)
        nodes_selected_positiv,indices_one_neighbour_positiv = Quali_Evaluator.get_neighbours(adj_t,source_node,target_positiv_selected)

        sparse_tensor_negativ,node_mapping_negativ = Quali_Evaluator.create_sparse_tensor(nodes_selected_negativ,adj_t)
        sparse_tensor_positiv,node_mapping_positiv = Quali_Evaluator.create_sparse_tensor(nodes_selected_positiv,adj_t)
        
        
        
        node_importance_negativ = Quali_Evaluator.calculate_ig_link_predictions(
            node_mapping_negativ,source_node,target_negativ_selected,nodes_selected_negativ,model,predictor,graph,sparse_tensor_negativ)
        node_importance_positiv = Quali_Evaluator.calculate_ig_link_predictions(
            node_mapping_positiv,source_node,target_positiv_selected,nodes_selected_positiv,model,predictor,graph,sparse_tensor_positiv)
        
        if one_hop:
            index_one_hopp_negativ = [idx in indices_one_neighbour_negativ for idx in nodes_selected_negativ]
            node_importance_negativ = node_importance_negativ[index_one_hopp_negativ]

            index_one_hopp_positiv = [idx in indices_one_neighbour_positiv for idx in nodes_selected_positiv]
            node_importance_positiv = node_importance_positiv[index_one_hopp_positiv]

            sparse_tensor_negativ,node_mapping_negativ = Quali_Evaluator.create_sparse_tensor(indices_one_neighbour_negativ,adj_t)
            sparse_tensor_positiv,node_mapping_positiv = Quali_Evaluator.create_sparse_tensor(indices_one_neighbour_positiv,adj_t)
            
            nodes_selected_negativ = indices_one_neighbour_negativ
            nodes_selected_positiv = indices_one_neighbour_positiv

        sparse_tensor_negativ = sparse_tensor_negativ.remove_diag()
        sparse_tensor_positiv= sparse_tensor_positiv.remove_diag()

        if (len(nodes_selected_negativ) > 500) or (len(nodes_selected_positiv) > 500):
            return
        low_dimensional_data_negativ, average_distance_negativ, maximal_distance_negativ = Quali_Evaluator.calc_low_dim_features(
            graph, nodes_selected_negativ)
        low_dimensional_data_positiv, average_distance_positiv, maximal_distance_positiv = Quali_Evaluator.calc_low_dim_features(
            graph, nodes_selected_positiv)
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))
        plt.suptitle(
            f'{prefix_title} Computational Graph with Node Importance', fontsize=28, y=1.1)

        # Plot negative visualization
        Quali_Evaluator.visualize_computational_graph(
            axs[0],
            low_dimensional_data_negativ,
            nodes_selected_negativ,
            np.array(torch.stack([sparse_tensor_negativ.storage.row(),sparse_tensor_negativ.storage.col()])).T,
            source_node,
            target_negativ_selected,
            average_distance_negativ,
            maximal_distance_negativ,
            mrr_list,
            position_selected,
            node_importance_negativ,
            "Highest probability wrong prediction",
            node_mapping_negativ
        )

        # Plot positive visualization
        Quali_Evaluator.visualize_computational_graph(
            axs[1],
            low_dimensional_data_positiv,
            nodes_selected_positiv,
            np.array(torch.stack([sparse_tensor_positiv.storage.row(),sparse_tensor_positiv.storage.col()])).T,
            source_node,
            target_positiv_selected,
            average_distance_positiv,
            maximal_distance_positiv,
            mrr_list,
            position_selected,
            node_importance_positiv,
            "Ground Truth Prediction",
            node_mapping_positiv
        )
        plt.tight_layout(pad=3.0)
        plt.savefig(r"./temp/computational_graph.png", bbox_inches='tight')
        experiment.log_image(r"./temp/computational_graph.png",
                             name=f"{prefix_title} Computational graph, source node: {source_node}")

        plt.close()
