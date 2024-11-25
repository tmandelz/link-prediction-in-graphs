import pandas as pd
import shutil
import os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw,\
                                    read_node_label_hetero, read_nodesplitidx_split_hetero,\
                                        read_binary_graph_raw, read_binary_heterograph_raw
from ogb.utils.torch_util import replace_numpy_with_torchtensor


class NodePropPredDataset(object):
    def __init__(self, name, root='dataset', meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  # original name, e.g., ogbn-proteins

        if meta_dict is None:
            # replace hyphen with underline, e.g., ogbn_proteins
            self.dir_name = '_'.join(name.split('-'))
            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(
                __file__), 'master_node.csv'), index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            # if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
            #     shutil.rmtree(self.root)

        # name of downloaded file, e.g., tox21
        self.download_name = self.meta_info['download_name']

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.num_classes = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(NodePropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed')

        if osp.exists(pre_processed_file_path):
            loaded_dict = torch.load(pre_processed_file_path)
            self.graph, self.labels = loaded_dict['graph'], loaded_dict['labels']

        else:
            # check download
            if self.binary:
                # npz format
                has_necessary_file_simple = osp.exists(
                    osp.join(self.root, 'raw', 'data.npz')) and (not self.is_hetero)
                has_necessary_file_hetero = osp.exists(
                    osp.join(self.root, 'raw', 'edge_index_dict.npz')) and self.is_hetero
            else:
                # csv file
                has_necessary_file_simple = osp.exists(
                    osp.join(self.root, 'raw', 'edge.csv.gz')) and (not self.is_hetero)
                has_necessary_file_hetero = osp.exists(
                    osp.join(self.root, 'raw', 'triplet-type-list.csv.gz')) and self.is_hetero

            has_necessary_file = has_necessary_file_simple or has_necessary_file_hetero

            if not has_necessary_file:
                url = self.meta_info['url']
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(osp.join(self.original_root,
                                self.download_name), self.root)
                else:
                    print('Stop download.')
                    exit(-1)

            raw_dir = osp.join(self.root, 'raw')

            # pre-process and save
            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(
                    ',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(
                    ',')

            if self.is_hetero:
                if self.binary:
                    self.graph = read_binary_heterograph_raw(
                        # only a single graph
                        raw_dir, add_inverse_edge=add_inverse_edge)[0]

                    tmp = np.load(osp.join(raw_dir, 'node-label.npz'))
                    self.labels = {}
                    for key in list(tmp.keys()):
                        self.labels[key] = tmp[key]
                    del tmp
                else:
                    self.graph = read_csv_heterograph_raw(
                        # only a single graph
                        raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files, additional_edge_files=additional_edge_files)[0]
                    self.labels = read_node_label_hetero(raw_dir)

            else:
                if self.binary:
                    self.graph = read_binary_graph_raw(raw_dir, add_inverse_edge=add_inverse_edge)[
                        0]  # only a single graph
                    self.labels = np.load(
                        osp.join(raw_dir, 'node-label.npz'))['node_label']
                else:
                    self.graph = read_csv_graph_raw(raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                                                    # only a single graph
                                                    additional_edge_files=additional_edge_files)[0]
                    self.labels = pd.read_csv(
                        osp.join(raw_dir, 'node-label.csv.gz'), compression='gzip', header=None).values

            print('Saving...')
            torch.save({'graph': self.graph, 'labels': self.labels},
                       pre_processed_file_path, pickle_protocol=4)

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(
                path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = train_idx_dict[nodetype]
                valid_idx_dict[nodetype] = valid_idx_dict[nodetype]
                test_idx_dict[nodetype] = test_idx_dict[nodetype]

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = pd.read_csv(
                osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
            valid_idx = pd.read_csv(
                osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
            test_idx = pd.read_csv(
                osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.labels

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  # original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(
                __file__), 'master_node.csv'), index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            # if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
            #     shutil.rmtree(self.root)

        # name of downloaded file, e.g., tox21
        self.download_name = self.meta_info['download_name']

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(PygNodePropPredDataset, self).__init__(
            self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(
                path)
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(
                path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(
                    train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(
                    valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(
                    test_idx_dict[nodetype]).to(torch.long)

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = torch.from_numpy(pd.read_csv(osp.join(
                path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(pd.read_csv(osp.join(
                path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(pd.read_csv(osp.join(
                path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')

    def download(self):
        url = self.meta_info['url']
        if True:
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root,
                        self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(
                ',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(
                ',')

        if self.is_hetero:
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(
                            node_label).to(torch.float32)
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(
                            node_label).to(torch.long)
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(
                        node_label).to(torch.float32)

        else:
            data = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                                  additional_edge_files=additional_edge_files, binary=self.binary)[0]

            # adding prediction target
            if self.binary:
                node_label = np.load(
                    osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(
                    self.raw_dir, 'node-label.csv.gz'), compression='gzip', header=None).values

            if 'classification' in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)

            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)

        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class PygLinkPropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  # original name, e.g., ogbl-ppa

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(
                __file__), 'master_link.csv'), index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            # if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
            #     shutil.rmtree(self.root)

        # name of downloaded file, e.g., ppassoc
        self.download_name = self.meta_info['download_name']

        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(PygLinkPropPredDataset, self).__init__(
            self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_edge_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train = replace_numpy_with_torchtensor(
            torch.load(osp.join(path, 'train.pt')))
        valid = replace_numpy_with_torchtensor(
            torch.load(osp.join(path, 'valid.pt')))
        test = replace_numpy_with_torchtensor(
            torch.load(osp.join(path, 'test.pt')))

        return {'train': train, 'valid': valid, 'test': test}

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')

    def download(self):
        url = self.meta_info['url']
        if True:
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root,
                        self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(
                ',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(
                ',')

        if self.is_hetero:
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files, binary=self.binary)[0]
        else:
            data = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                                  additional_edge_files=additional_edge_files, binary=self.binary)[0]

        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
