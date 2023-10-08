import pandas as pd
from torch_geometric.data import Data
import json


# format json -> csv
df = pd.DataFrame()
json_file_path = 'facebook_large/musae_facebook_features.json'
with open(json_file_path, 'r') as json_file:
    features_dict = json.load(json_file)
    
    for key, item in features_dict.items():
        # df = df.append({key: item}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame({key: item})], axis=0)
    
    df.to_csv('musae_facebook_features.csv', index=False)
    # print(df)
