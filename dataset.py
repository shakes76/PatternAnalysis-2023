import pandas as pd
from torch_geometric.data import Data
import json


# format json -> csv
def json_to_csv():
    df = pd.DataFrame()
    json_file_path = 'facebook_large/musae_facebook_features.json'
    
    with open(json_file_path, 'r') as json_file:
        features_dict = json.load(json_file)

        for key, item in features_dict.items():
            df1 = pd.DataFrame({'key':key, 'item': item})
            df = pd.concat([df, df1], ignore_index=True)

    df.to_csv('facebook_large/musae_facebook_features.csv', index=False)

