import shapely#.geometry import Point, LineString
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')



class GeoPath():

    def __init__(self, sending_id, recieving_id, gdf, df = None, degrees = 1):


        """
        TO-DO EDGE CASES:
            1) When shortest path crosses a body of water (or another countries border) need to reroute to shortest path that stays on land/in-country
            2) Need to prevent municiaplties that are too far north/south of sending/recieivng from being included in neighbor list
                - I think this is going to involve calculating a trajectory (i.e.. are they traveling north/south) and then maybe buffering the relavent sides of sending/recieiving                    nodes to prevent muni's within the buffer from being included
        """

        self.df = df
        self.gdf = gdf
        self.degrees = degrees
        self.sending_id = sending_id
        self.recieving_id = recieving_id
        self.shortest_distance_nodes = self.__get_shortest_distance_nodes()
        self.neighbors_gdf, self.neighbors = self.__get_spatial_neighbors()
        self.neighbors_recoded = self.__index_neighbors()
        self.edge_list = self.__make_edge_list()
        self.adj_list = self.__make_adj_list()
        if df is not None:
            self.x = self.__get_x()
            self.y = self.__get_y()


    def __calc_centroid_x(self, x):
        return x.centroid.x
    
    def __calc_centroid_y(self, x):
        return x.centroid.y

    def calc_shortest_line(self, xs, ys):
        line = shapely.geometry.LineString([shapely.geometry.Point(xs[0], ys[0]), shapely.geometry.Point(xs[1], ys[1])]).wkt
        new_row = pd.DataFrame(columns = ['shapeID', 'geometry', 'c_long', 'c_lat'])
        new_row['shapeID'], new_row['geometry'], new_row['c_long'], new_row['c_lat'] = ['line'], [line], [0], [0]
        new_row['geometry'] = new_row['geometry'].apply(shapely.wkt.loads)
        new_row = gpd.GeoDataFrame(new_row, geometry='geometry')
        return new_row


    def __get_intersection(self):
        inp, res = self.gdf.sindex.query_bulk(self.line.geometry, predicate='intersects')
        intersected = self.gdf.loc[res]
        return intersected


    def __get_shortest_distance_nodes(self):
        gdf_temp = self.gdf[self.gdf['shapeID'].isin([self.sending_id, self.recieving_id])]
        gdf_temp['c_long'] = gdf_temp['geometry'].apply(lambda x: self.__calc_centroid_x(x))
        gdf_temp['c_lat'] = gdf_temp['geometry'].apply(lambda x: self.__calc_centroid_y(x))
        self.line = self.calc_shortest_line(gdf_temp['c_long'].to_list(), gdf_temp['c_lat'].to_list()).buffer(.001)
        intersected = self.__get_intersection()
        return intersected


    def __make_neighbor_list(self, x):
        return self.gdf[~self.gdf.geometry.disjoint(x)].shapeID.tolist()


    def __get_spatial_neighbors(self):

        self.degree_dict = {}
        neighbors_gdf = self.shortest_distance_nodes
        neighbors_gdf = neighbors_gdf[~neighbors_gdf['shapeID'].isin([self.sending_id, self.recieving_id])]

        self.degree_dict[0] = [self.sending_id, self.recieving_id]
        self.degree_dict[1] = self.shortest_distance_nodes['shapeID'].to_list()

        for i in range(self.degrees - 1):

            neighbors_gdf['neighbors'] = neighbors_gdf.geometry.apply(lambda x: self.__make_neighbor_list(x))
            all_neighbors = dict(zip(neighbors_gdf['shapeID'], neighbors_gdf['neighbors']))

            u_vals = list(set([item for sublist in all_neighbors.values() for item in sublist]))
            all_in_degree_dict = list(set([item for sublist in self.degree_dict.values() for item in sublist]))
            new_for_degree_dict = [i for i in u_vals if i not in all_in_degree_dict]
            self.degree_dict[i + 2] = new_for_degree_dict

            remove_vals = [i for i in u_vals if i not in all_neighbors.keys()]
            for k,v in all_neighbors.items():
                to_remove = [j for j in v if j in remove_vals]
                for tr in to_remove:
                    all_neighbors[k] = [i for i in all_neighbors[k] if i not in tr]


            all_munis = list(all_neighbors.keys()) + u_vals
            neighbors_gdf = self.gdf[self.gdf['shapeID'].isin(all_munis)]
            neighbors_gdf = neighbors_gdf[~neighbors_gdf['shapeID'].isin([self.sending_id, self.recieving_id])]

        to_append = self.shortest_distance_nodes[self.shortest_distance_nodes['shapeID'].isin([self.sending_id, self.recieving_id])]
        neighbors_gdf = neighbors_gdf.append(to_append)

        return neighbors_gdf, all_neighbors


    def __get_x(self):
        """
        Returns a numpy array of shape(len(self.neighbors.keys()), len(self.neighbors.keys())) storing
        the feature data for all of the municipaities in the graph in the order self.neighbors.keys()
        """
        all_x = []
        for k in self.neighbors.keys():
            cur_data = self.df[self.df['shapeID'] == k].values
            if len(cur_data) != 0:
                all_x.append(cur_data[0][1:-1]) 
            else:
                all_x.append(np.array([0] * 4))

        return np.array(all_x, dtype = np.float32)


    def __get_y(self):
        """
        Returns the y value of the target municipality if there is data, otherwise returns 0
        OBVIOUSLY NEED TO GO BACK AND FIX THIS
        """
        return random.randint(0, 1000)
        # y = self.df[self.df['shapeID'] == self.target_id]['sum_num_intmig'].values
        # if len(y) == 0:
        #     return np.array([0])
        # else:
        #     return y


    def __index_neighbors(self):
        """
        Replaces each municiapaity shapeID string with a unique index uin range(0, len(self.neighbors.keys()). 
        Format & ordering is the exact same as self.neighbirs
        """
        neighbors_ref_dict = dict(zip(self.neighbors.keys(), [i for i in range(len(self.neighbors.keys()))]))
        new_neighbors = {}
        for k in self.neighbors.keys():
            new_neighbors[neighbors_ref_dict[k]] = [neighbors_ref_dict[i] for i in self.neighbors[k]]
        return new_neighbors


    def __make_edge_list(self):
        edge_list = []
        for k,v in self.neighbors_recoded.items():
            [edge_list.append([k, cur_v]) for cur_v in v]
        return edge_list


    def __make_adj_list(self):
        """
        Returns an adjacency list based on the self.neighbors_recoded dictionary.
        Since every elemnt in the array needs to have the same number of values, but muni's don't all 
        have the same number of neighbors, it fills the remaining elements in each list with the value -99.
        """
        max_n = len(self.neighbors_recoded.keys())
        adj_list = np.full((max_n, max_n), -99)
        for i in self.neighbors_recoded.values():
            cur_new_vals = np.pad(np.array(i), (0, max_n - len(i)), constant_values = -99)
            try:
                new_values = np.concatenate((new_values, cur_new_vals))
            except Exception as e:
                new_values = cur_new_vals
        return np.reshape(new_values, (max_n, max_n))


    def __map_degree_column(self, x):
        """
        Returns the number of degrees away from the target_id a municipality is
        """
        for k,v in self.degree_dict.items():
            if x in v:
                return k


    def show(self, adm0 = None):

        self.neighbors_gdf['degree'] = self.neighbors_gdf['shapeID'].apply(lambda x: self.__map_degree_column(x))

        if adm0 is None:

            plt.figure(1, figsize=(10, 10)) 
            self.neighbors_gdf.plot(column = 'degree', cmap = 'viridis')
            plt.title(self.sending_id + "->" + self.recieving_id + "\n Degrees: " + str(self.degrees))

        else:

            plt.figure(1, figsize=(10, 10)) 
            base = adm0.plot()
            self.neighbors_gdf.plot(ax = base, column = 'degree', cmap = 'viridis')
            plt.title(self.sending_id + " -> \n" + self.recieving_id + "\n Degrees: " + str(self.degrees))


    def __str__(self):
        if self.df is not None:
            return 'GeoPath(x = [' + str(self.x.shape[0]) + "," + str(self.x.shape[1]) + "], adj_list = [" + str(self.adj_list.shape[0]) + "," + str(self.adj_list.shape[1]) + "], y = " + str(self.y) + ")" 
        else:
            return "GeoPath(adj_list = [" + str(self.adj_list.shape[0]) + "," + str(self.adj_list.shape[1]) + "])" 



if __name__ == "__main__":

    # gdf = gpd.read_file("./MEX/MEX_ADM2_fixedInternalTopology.shp")
    gdf = gpd.read_file("./yemen/geoBoundariesSimplified-3_0_0-YEM-ADM2-shp/geoBoundariesSimplified-3_0_0-YEM-ADM2.shp")
    gdf = gdf[['shapeID', 'geometry']]

    # match = pd.read_csv("./gB_IPUMS_match.csv")
    # match = match[['shapeID', 'MUNI2015']]
    # ref_dict = dict(zip(match['MUNI2015'], match['shapeID']))

    # df = pd.read_csv("./mexico2010.csv")
    # df = df[['GEO2_MX', 'sum_income', 'total_pop', 'unrel_ppl', 'perc_urban', 'sum_num_intmig']]
    # df['GEO2_MX'] = df['GEO2_MX'].astype(str).str.replace("484", "").astype(int).map(ref_dict)
    # df = df.rename(columns = {'GEO2_MX': 'shapeID'})

    sending_id = random.choice(gdf['shapeID'].to_list())
    recieving_id = random.choice(gdf['shapeID'].to_list())

    degrees = random.randint(1, 4)

    # adm0 = gpd.read_file("./geoBoundariesSimplified-3_0_0-MEX-ADM0-shp/geoBoundariesSimplified-3_0_0-MEX-ADM0.shp")
    adm0 = gpd.read_file("./yemen/geoBoundariesSimplified-3_0_0-YEM-ADM0-shp/geoBoundariesSimplified-3_0_0-YEM-ADM0.shp")

    GeoPath(sending_id, recieving_id, gdf, df = None, degrees = degrees).show(adm0 = adm0)
    plt.savefig("./test_path.png")
