import geograph as gg
import numpy as np
import random
import torch
import os


class ImageGraphLoader():

    class Dataset():
        def __init__(self, x, y, edge_index, neighbors, adj_matrix, batch):
            self.x = x
            self.y = y
            self.edge_index = edge_index
            self.neighbors = neighbors
            self.adj_matrix = adj_matrix
            self.batch = batch

    def __init__(self, data_dir, iso, dta_path, batch_size):

        self.iso = iso
        self.data_dir = os.path.join(data_dir, iso)
        self.batch_size = batch_size
        self.munis = [i for i in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, i))]
        self.data_dir = os.path.join(data_dir, iso)
        self.image_graphs = [gg.ImageGraph(self.data_dir, i, dta = dta_path) for i in self.munis] 
        self.indexes = [i for i in range(0, len(self.munis))]
        random.shuffle(self.indexes)
        self.indexes = [self.indexes[i:i + self.batch_size] for i in range(0, len(self.indexes), self.batch_size)]

        self.data = []
        [self.data.append(self.__load_graph(np.array(self.image_graphs)[batch])) for batch in self.indexes]

    
    def __make_adj_matrix(self, edge_list, dim):
        adj_matrix = np.zeros((dim, dim))
        for edge in edge_list:
            adj_matrix[edge[0]][edge[1]] = 1
        for i in range(dim):
            adj_matrix[i][i] = 1
        return adj_matrix


    def __load_graph(self, batch):

        # Neighbors
        node_nums = [i.num_nodes for i in batch]
        neighbors = [i.neighbors for i in batch]
        keys, vals = [], []
        for i in range(len(node_nums)):
            keys.append(np.array(list(neighbors[i].keys())) + np.sum(np.array(node_nums)[:i]))
            vals_list = list(neighbors[i].values())
            [vals.append(v + np.sum(np.array(node_nums)[:i])) for v in vals_list]
        keys = [i.tolist() for i in keys]
        keys = [item for sublist in keys for item in sublist]
        vals = [i.tolist() for i in vals]
        neighbors = dict(zip(keys, vals))

        # X's 
        xs = torch.cat([i.x for i in batch])

        # Y's
        ys = torch.tensor([i.y for i in batch]).view(-1, 1)

        # Batch ID's
        batch_ids = []
        for i in range(len(node_nums)):
            batch_ids.append(np.array([i for n in range(node_nums[i])], dtype = np.float32))        
        batch_ids = torch.tensor(np.concatenate(batch_ids))

        # Edge Indices
        all_edge_indices = []
        edge_indices = [i.edge_list for i in batch]
        for i in range(len(edge_indices)):
            all_edge_indices.append(np.array(edge_indices[i]) + np.sum(np.array(node_nums)[:i]))
        edge_indices = torch.tensor(np.concatenate(all_edge_indices))

        adj_matrix = self.__make_adj_matrix(edge_indices, len(neighbors))
        
        return self.Dataset(xs, ys, edge_indices, neighbors, adj_matrix, batch_ids)