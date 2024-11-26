from torch_geometric.data import InMemoryDataset
import numpy as np
import torch
import os
import random
import sys
from ogb.linkproppred import PygLinkPropPredDataset
sys.path.append('modelling/')
from dataset_split.dataset_loader import PygLinkPropPredDataset, PygNodePropPredDataset

class Dataset_Splitter:
    def __init__(self,sample_size) -> None:
        self.sample_size = sample_size
        pass

    def load_dataset(self, dataset_name: str) -> dict:
        """loads a Link or Node Property Prediction Dataset

        Args:
            dataset_name (str): name of the dataset to load

        Returns:
            dict: dataset as a dictionary
        """
        # set the correct dataset type
        if dataset_name == 'ogbl-citation2':
            dataset = PygLinkPropPredDataset(name=dataset_name)
        elif dataset_name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name=dataset_name)
        elif dataset_name == 'ogbn-papers100M':
            dataset = PygNodePropPredDataset(name=dataset_name)
        else:
            raise ValueError(
                f"Dataset {dataset_name} does not exist or is not implemented.")

        return dataset

    def node_index_train_eval_papers100M(self, graph_data: dict, year: int = 2018) -> tuple:
        """ returns the node indexes for training and for evaluation data for papers 100 M Dataset
        uses only the year 2018 to ensure no data leakage problems for the eval sets of the other datasets

        Args:
            graph_data (dict): graph dictionary
            year (int, optional): year to separate training and eval data. Defaults to 2018.

        Returns:
            tuple: index for training nodes, index for evaluation nodes
        """
        # defines indexes as train or eval
        index_train = torch.nonzero(
            graph_data.node_year.reshape(-1) < year).squeeze()
        index_eval = torch.nonzero(
            graph_data.node_year.reshape(-1) == year).squeeze()
        return index_train, index_eval

    def node_index_train_eval(self, graph_data: dict, year: int = 2019) -> tuple:
        """returns the node indexes for training and for evaluation data

        Args:
            graph_data (dict): graph dictionary
            year (int, optional): year to separate training and eval data. Defaults to 2019.

        Returns:
            tuple: index for training nodes, index for evaluation nodes
        """
        # defines indexes as train or eval
        index_train = torch.nonzero(
            graph_data.node_year.reshape(-1) < year).squeeze()
        index_eval = torch.nonzero(
            graph_data.node_year.reshape(-1) >= year).squeeze()
        return index_train, index_eval

    def get_negative_edges(self, edge_index: torch.tensor, eval_edges: list, sample_size: int = 1000) -> torch.tensor:
        """samples not existing edges for the eval sets, sampled edges can also be nodes from the train set

        Args:
            edge_index (torch.tensor): all edges of the dataset
            eval_edges (list): edges for the val or test set
            sample_size (int, optional): sample size to generate. Defaults to 1000.

        Returns:
            torch.tensor: all sampled negative edges
        """
        # take only the source nodes of the eval set
        negative_edges_pairs_source_nodes = torch.unique(
            torch.tensor(eval_edges).T[0]).numpy()

        # init sampled negatives list
        sampled_negatives_list = []

        num_nodes = int(torch.max(edge_index).item()) + 1
        # Create adjacency matrix from edge index
        indices = edge_index
        # Assuming all edges have weight 1
        values = torch.ones(edge_index.shape[1])
        adj_matrix = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes))
        adj_matrix = adj_matrix.coalesce()
        # Initialize sampled negatives list
        sampled_negatives_list = []

        # For each node in the eval set, sample non-existing references
        for node_of_interest in negative_edges_pairs_source_nodes:
            # Get neighbors of the eval node
            row_indices = adj_matrix.indices()[0]
            mask = row_indices == node_of_interest
            neighbors = adj_matrix.indices()[1][mask]

            # Find nodes not neighbors (they don't have an edge)
            all_nodes = torch.arange(num_nodes)
            not_neighbors = all_nodes[torch.logical_not(
                torch.isin(all_nodes, neighbors))]

            # Ensure node_of_interest is excluded
            not_neighbors = not_neighbors[not_neighbors != node_of_interest]

            # Sample k not existing edges per node
            sampled_nodes = not_neighbors[random.sample(
                range(not_neighbors.size(0)), sample_size)]
            sampled_negatives_list.append(sampled_nodes.tolist())
        # create a tensor from the sampled edges
        sampled_negatives_list = torch.tensor(np.array(sampled_negatives_list))
        return sampled_negatives_list

    def sample_eval_edges(self, edge_index: torch.tensor, index_eval: torch.tensor) -> tuple:
        """For each eval paper, randomly select two papers from its references for validation and testing

        Args:
            edge_index (torch.tensor): all edges of the dataset
            index_eval (torch.tensor): index of the nodes in the evaluation set

        Returns:
            tuple: (validation edges sampled, test edges sampled)
        """
        validation_edges_sampled = []
        test_edges_sampled = []
        all_eval_papers = torch.unique(edge_index[:, index_eval][0])
        # for each evaluation paper node id sample two references, one is validation and the other is for testing
        for eval_paper_node_id in all_eval_papers:
            # all references of the eval paper
            eval_paper_edges = edge_index[:,
                                          edge_index[0] == eval_paper_node_id]
            references_count = eval_paper_edges.shape[1]

            if references_count >= 2:
                # Randomly select two references from the paper
                indices = np.random.choice(
                    references_count, size=2, replace=False)
                # get sampled edges, first is val set, second is test set
                sampled_edges = eval_paper_edges[:, indices]
                validation_edges_sampled.append(sampled_edges[:, 0])
                test_edges_sampled.append(sampled_edges[:, 1])
            else:
                # if paper has not at least two references for val and test set continue and ignore this paper
                continue

        return validation_edges_sampled, test_edges_sampled

    def create_split_edges(self, edge_index: torch.tensor, index_eval: torch.tensor) -> tuple:
        """creates the split edge by separating all edges into train and eval sets.

        Args:
            edge_index (torch.tensor): all edges of the dataset
            index_eval (torch.tensor): index of the nodes in the evaluation set
            eval_node_count (int): count if nodes to select for the eval set

        Returns:
            tuple: (split_edge_train, split_edge_valid, split_edge_test)
        """
        validation_edges_sampled, test_edges_sampled = self.sample_eval_edges(
            edge_index, index_eval)

        # Convert lists to tensors
        validation_edges_sampled = torch.stack(validation_edges_sampled, dim=1)
        test_edges_sampled = torch.stack(test_edges_sampled, dim=1)

        # sample negative edges for valid and test set
        target_nodes_neg_valid = self.get_negative_edges(edge_index, list(
            zip(validation_edges_sampled[0].tolist(), validation_edges_sampled[1].tolist())),self.sample_size)
        target_nodes_neg_test = self.get_negative_edges(edge_index, list(
            zip(test_edges_sampled[0].tolist(), test_edges_sampled[1].tolist())),self.sample_size)

        # Separate the remaining edges for training
        # Concatenate validation_pairs and test_pairs along the second dimension (columns)
        validation_test_pairs = torch.cat(
            (validation_edges_sampled, test_edges_sampled), dim=1)
        # find all pairs that match in row 1 and also in row 2
        # all edges which are in the validation or test pairs should not be included in the train set, everything else is included
        matching_pairs = (torch.isin(edge_index[0], validation_test_pairs[0]) &
                          torch.isin(edge_index[1], validation_test_pairs[1]))
        # reverse the mask to get all other node pairs which were not found in the validation testing
        negation_matching_pairs = ~matching_pairs
        train_pairs = edge_index[:, negation_matching_pairs]

        # put dictionaries together for the split edges
        split_edge_train = {"source_node": train_pairs[0],
                            "target_node": train_pairs[1]}
        split_edge_valid = {"source_node": validation_edges_sampled[0],
                            "target_node": validation_edges_sampled[1],
                            "target_node_neg": target_nodes_neg_valid}
        split_edge_test = {"source_node": test_edges_sampled[0],
                           "target_node": test_edges_sampled[1],
                           "target_node_neg": target_nodes_neg_test}

        return split_edge_train, split_edge_valid, split_edge_test

    def get_edges_split(self, dataset: InMemoryDataset) -> tuple:
        """returns the dataset splits according to our split rules

        Args:
            dataset (InMemoryDataset): takes a InMemorydataset from ogb (PygLinkPropPredDataset or PygNodePropPredDataset)

        Returns:
            tuple: (split edges containing train, val and test, edge_index )
        """
        data = dataset[0]
        edge_index = data.edge_index
        split_edge_file_path = fr"./temp/split_edge-{dataset.name}.pt"
        # if dataset is Node Classification -> change the splits like in the citation2 splits for link prediction
        if isinstance(dataset, PygNodePropPredDataset):
            print(f'Creating Splits for dataset {dataset.name}')

            if os.path.exists(split_edge_file_path):
                split_edge = torch.load(split_edge_file_path)
            else:
                # get eval nodes
                if dataset.name == "ogbn-papers100M":
                    _, index_eval = self.node_index_train_eval_papers100M(data)
                else:
                    _, index_eval = self.node_index_train_eval(data)
                # splits
                split_edge_train, split_edge_valid, split_edge_test = self.create_split_edges(
                    edge_index, index_eval)
                # put splits together
                split_edge = {"train": split_edge_train,
                              "valid": split_edge_valid,
                              "test": split_edge_test,
                              }
                # create folder if not exists
                os.makedirs("./temp/", exist_ok=True)
                torch.save(split_edge, split_edge_file_path)
        # if dataset is Link Prediction -> return split edge
        elif isinstance(dataset, PygLinkPropPredDataset):
            split_edge = dataset.get_edge_split()
        print(f'Returning Splits for dataset {dataset.name}')
        return split_edge, edge_index
