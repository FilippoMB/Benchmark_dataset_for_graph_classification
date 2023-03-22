import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from os import path
import numpy as np


class GraphClassificationBench(InMemoryDataset):
    """The synthetic dataset from `"Pyramidal Reservoir Graph Neural Network"
    <https://arxiv.org/abs/2104.04710>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If `"train"`, loads the training dataset.
            If `"val"`, loads the validation dataset.
            If `"test"`, loads the test dataset. Defaults to `"train"`.
        easy (bool, optional): If `True`, use the easy version of the dataset.
            Defaults to `True`.
        small (bool, optional): If `True`, use the small version of the
            dataset. Defaults to `True`.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to `None`.
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. Defaults to `None`.
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. Defaults to `None`.
    """
    base_url = ('http://github.com/FilippoMB/'
                'Benchmark_dataset_for_graph_classification/'
                'raw/master/datasets/')
    
    def __init__(self, root, split='train', easy=True, small=True, transform=None, pre_transform=None, pre_filter=None):
        self.split = split.lower()
        assert self.split in {'train', 'val', 'test'}
        if self.split != 'val':
            self.split = self.split[:2]
        
        self.file_name = ('easy' if easy else 'hard') + ('_small' if small else '')
        
        super(GraphClassificationBench, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return '{}.npz'.format(self.file_name)
    
    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.file_name+ '_' + self.split)

    def download(self):
        download_url('{}{}.npz'.format(self.base_url, self.file_name), self.raw_dir)

    def process(self):
        npz = np.load(path.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
        raw_data = (npz['{}_{}'.format(self.split, key)] for key in ['feat', 'adj', 'class']) 
        data_list = [Data(x=torch.FloatTensor(x), 
                          edge_index=torch.LongTensor(np.stack(adj.nonzero())), 
                          y=torch.LongTensor(y.nonzero()[0])) for x, adj, y in zip(*raw_data)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


if __name__ == "__main__":
    file_path = "data/"
    tr_dataset = GraphClassificationBench(file_path, split='train', easy=False, small=False)
    print(tr_dataset[5])