import pandas as pd
import torch
from torch_geometric.data import Data
import json
import networkx as nx
import matplotlib.pyplot as plt



def json_to_csv(json_file_path='facebook_large/musae_facebook_features.json'):
    '''
    format json -> csv
    
    Returns: csv
    '''
    df = pd.DataFrame()
    
    with open(json_file_path, 'r') as json_file:
        features_dict = json.load(json_file)

        for key, item in features_dict.items():
            df1 = pd.DataFrame({'key':key, 'item': item})
            df = pd.concat([df, df1], ignore_index=True)

    df.to_csv('facebook_large/musae_facebook_features.csv', index=False)

def format_target(csv_file_path='facebook_large/musae_facebook_target.csv'):
    df = pd.read_csv(csv_file_path).iloc[:, -1]
    df.columns = ['index', 'target']
    df.to_csv('facebook_large/musae_facebook_target1.csv', index=True)

format_target()

# features = pd.read_csv('facebook_large/musae_facebook_features.csv')
# x = torch.Tensor(features.values)

# edge_index = pd.read_csv('facebook_large/musae_facebook_edges.csv')
# y = pd.read_csv('facebook_large/musae_facebook_target.csv')

# data = Data(x=x, edge_index=edge_index, y=y)

# print(data.is_node_attr)


edge_list_df = pd.read_csv('facebook_large/musae_facebook_edges.csv')
edge_list = nx.from_pandas_edgelist(edge_list_df, source='id_1', target='id_2')
