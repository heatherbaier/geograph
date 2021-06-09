from torchvision import transforms, utils
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio as rio
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import zipfile
import shapely
import random
import shutil
import torch
import json
import os

from rasterio import plot
from rasterio.plot import show


class ImageGraph():

    def __init__(self, imagery_dir, adm_id, dta = None):

        """
        Args:
            - target_id: If loading data for a municiaplity shapefile, should be the 
              unique shapeID of a municipality, otherwise use 'search' to find the central 
              most node in a shapefile (use this when you load in the imagery boxes)
            - gdf: dataframe with geometry and x & y data IF load_data == True
            - degrees: number of degrees away fromt he target municiaplity to contruct the graph
        __init__ variables:
            - degree_dict: dictionary with keys 0...self.degrees with the values being the list of 
              shapeID's that are k degrees from the target
            - neighbors: dictionary with the keys being each of the municiaplites within self.degrees
              from the target and the values being that municiaplites neighbors (that are no further th)
        """

        self.adm_id = adm_id 
        self.target_path = os.path.join(imagery_dir, adm_id)
        self.imagery_dir = os.path.join(self.target_path, "pngs")
        self.zip_path = os.path.join(self.target_path, "imagery")
        self.temp_path = os.path.join(self.target_path, "temp")
        self.shp_path = os.path.join(self.target_path, [i for i in os.listdir(self.target_path) if i.endswith(".shp")][0])
        self.gdf = gpd.read_file(self.shp_path)
        self.degree_dict = {}
        self.target_id = 0
        self.degrees = 100
        

        self.x = self.__load_imagery()
        self.neighbors = self.__get_spatial_neighbors()
        self.edge_list = self.__make_edge_list()
        self.adj_list = self.__make_adj_list()
        self.adj_matrix = self.__make_adj_matrix()
        self.num_nodes = len(self.neighbors.keys())

        if dta is not None:
            self.dta_path = dta
            self.y = self.__get_y()

    def __load_image(self, image_path):
        image_path = os.path.join(self.imagery_dir, image_path)
        to_tens = transforms.ToTensor()
        return to_tens(Image.open(image_path).convert('RGB')).unsqueeze(0)

    def __load_imagery(self):
        images = os.listdir(self.imagery_dir)
        images = torch.cat([self.__load_image(i) for i in list(images)], dim = 0)
        return images

    def __get_spatial_neighbors(self):
        """
        - Returns a dictionary with the keys being the shapeID's of the municipalities in the graph 
          (within self.degrees) and the values being the neighbors of the shapeID key
        - Runs on initialization
        """
        row = self.gdf[self.gdf['shapeID'] == self.target_id].squeeze()
        target_neighbors = self.gdf[~self.gdf.geometry.disjoint(row.geometry)].shapeID.tolist()
        neighbors = target_neighbors

        all_neighbors = {}
        self.degree_dict[0] = [self.target_id]
        self.degree_dict[1] = [i for i in target_neighbors if i != self.target_id]
    
        # Get neighbors
        for i in range(self.degrees):
            new_n = []
            for n in neighbors:
                cur_row = self.gdf[self.gdf['shapeID'] == n].squeeze()
                cur_neighbors = self.gdf[~self.gdf.geometry.disjoint(cur_row.geometry)].shapeID.tolist()
                if n not in all_neighbors.keys():
                    all_neighbors[n] = cur_neighbors
                    new_n.append(n)
            if i != 0:
                self.degree_dict[i + 1] = new_n

            k = [v for k,v in all_neighbors.items()]
            k = list(set([item for sublist in k for item in sublist]))
            k = [i for i in k if i not in all_neighbors.keys()]
            neighbors = k

            if len(neighbors) == 0:
                break

        # Cleanup: remove all ofthe neighbors of neighbors that are more than one degree fromt he target node
        # i.i. remove all of the muiciaplites in the values that are not in the keys
        u_vals = list(set([item for sublist in all_neighbors.values() for item in sublist]))
        remove_vals = [i for i in u_vals if i not in all_neighbors.keys()]
        for k,v in all_neighbors.items():
            to_remove = [j for j in v if j in remove_vals]
            for tr in to_remove:
                all_neighbors[k] = [i for i in all_neighbors[k] if i not in tr]

        return all_neighbors

    def __make_edge_list(self):
        edge_list = []
        for k,v in self.neighbors.items():
            [edge_list.append([k, cur_v]) for cur_v in v]
        return edge_list

    def __make_adj_list(self):
        """
        Returns an adjacency list based on the self.neighbors_recoded dictionary.
        Since every element in the array needs to have the same number of values, but muni's don't all 
        have the same number of neighbors, it fills the remaining elements in each list with the value -99.
        """
        max_n = len(self.neighbors.keys())
        adj_list = np.full((max_n, max_n), -99)
        for i in self.neighbors.values():
            cur_new_vals = np.pad(np.array(i), (0, max_n - len(i)), constant_values = -99)
            try:
                new_values = np.concatenate((new_values, cur_new_vals))
            except Exception as e:
                new_values = cur_new_vals
        return np.reshape(new_values, (max_n, max_n))

    def __make_adj_matrix(self):
        adj_matrix = np.zeros((len(self.gdf), len(self.gdf)))
        for edge in self.edge_list:
            adj_matrix[edge[0]][edge[1]] = 1
        for i in range(len(self.gdf)):
            adj_matrix[i][i] = 1
        return adj_matrix


    def __get_y(self):
        m = open(self.dta_path,)
        data = json.load(m)
        m.close()
        return data[self.adm_id]


    def show(self):

        try:
            os.mkdir(self.temp_path)
        except:
            shutil.rmtree(self.temp_path)
        
        for zipfolder in os.listdir(self.zip_path):
            with zipfile.ZipFile(os.path.join(self.zip_path, zipfolder), 'r') as zip_ref:
                zip_ref.extractall(self.temp_path)

        b1s = [i for i in os.listdir(self.temp_path) if i.endswith("B1.tif")]
        b1s = [rio.open(os.path.join(self.temp_path, i)) for i in b1s]

        fig, ax = plt.subplots(figsize = (12, 10))
        for i in b1s:
            show(i, ax = ax, transform = i.transform, cmap = 'gist_earth')
        self.gdf.plot(ax = ax, color = 'black', alpha = 0) ## alpha is the transparency setting
        # plt.show()

        shutil.rmtree(self.temp_path)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("test", help="Path to exported Mexico ADM2 shapefile")
    args = parser.parse_args()

    if args.test == 'random':

        adms = [i for i in os.listdir("./data/") if "-B" in i]
        print("ADM's available: ", adms)
        index = random.randint(0, len(adms) - 1)
        adm_id = adms[index]

        print("Selected ADM: ", adm_id)

        ig = ImageGraph(adm_id = adm_id)
        ig.show()
        plt.savefig("./test.png")

    else:

        import landsat_prep as lp

        GB_PATH = "./data/MEX/MEX_ADM2_fixedInternalTopology.shp"
        ADM_ID = args.test
        ISO = "MEX"
        IC = "LANDSAT/LT05/C01/T1"

        lp.prep_landsat(GB_PATH, ISO, ADM_ID, "2010", "1", IC)

        ig = ImageGraph(adm_id = ADM_ID)
        ig.show()
        plt.savefig("./test.png")