import csv
import numpy as np
from scipy.linalg import sqrtm
import json
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Create Adjacency matrix from the musae_facebook_edges.csv file"""
def create_adjacency_matrix():
    data = []

    # Read the CSV data from musae_facebook_edges.csv file
    with open('facebook_large/musae_facebook_edges.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        data = list(reader)

    # Determine all unique nodes, it will convert it to a list and sort it, then create a mapping of node to its index
    nodes = set()
    for row in data:
        nodes.add(row[0])
        nodes.add(row[1])
    nodes = sorted(list(nodes), key=int)  
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Initialize adjacency matrix with zeros
    matrix_size = len(nodes)
    adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Fill the matrix
    for row in data:
        i, j = node_to_index[row[0]], node_to_index[row[1]]
        adjacency_matrix[i][j] = 1
        # Because the links are symmetric
        adjacency_matrix[j][i] = 1  

    # Every node links to itself so the following must be done
    np.fill_diagonal(adjacency_matrix, 1)
    return adjacency_matrix

"""Normalise adjanceny matrix. Where D is the Degree matrix"""
def normalise_adjacency_matrix(adjacency_matrix):
    D = np.zeros_like(adjacency_matrix)
    np.fill_diagonal(D, np.asarray(adjacency_matrix.sum(axis=1)).flatten())
    D_invsqrt = np.linalg.inv(sqrtm(D))
    adjacency_normed = D_invsqrt @ adjacency_matrix @ D_invsqrt
    return adjacency_normed

adjacency_normed = normalise_adjacency_matrix(create_adjacency_matrix())
#print(adjacency_normed)


"""Since the feature .json file has an inconsistent number of features for each node.
 We need to make them consistent for training.
 Hence I will create an n-dimensional bag of words feature vector for each node"""
def create_feature_vectors():
    # Load data from a JSON file
    with open('facebook_large/musae_facebook_features.json', 'r') as file:
        data = json.load(file)

    # Convert string keys to integers
    data = {int(k): v for k, v in data.items()}

    # Get unique features
    all_features = set()
    for features in data.values():
        all_features.update(features)
    all_features = sorted(list(all_features))

    # Create n-dimensional bag of words feature vector for each node
    feature_list = []
    for node, features in data.items():
        vector = [node] + [1 if feature in features else 0 for feature in all_features]
        feature_list.append(vector)

    # Convert to 2D numpy array
    feature_vectors = np.array(feature_list, dtype=int)
    return feature_vectors

feature_vectors = create_feature_vectors()
# print(feature_vectors)



"""I will create a mapper function for the class labels since they are in string format.
 0 = politician, 1 = governmental orginisations, 2 = television shows and 3 = companies"""
def mapper(x):
    if x == "politician":
        y = 0
    elif x == "government":
        y = 1
    elif x == "tvshow":
        y = 2
    elif x == "company":
        y = 3
    return y

"""Labels in musae_facebook_target.csv are in string format thus we will convert them to an integer value so they can be used with the model"""
def convert_labels():
    processed_data = []

    # Read the CSV data
    with open('facebook_large/musae_facebook_target.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            id_ = int(row['id'])
            page_type = mapper(row['page_type'])
            processed_data.append([id_, page_type])

    # Convert the processed data into a numpy array
    node_labels = np.array(processed_data, dtype=int)
    return node_labels

node_labels = convert_labels()
# print(node_labels[:5])



""" t-SNE plot created to visualize the initial, high-dimensional node features in a 2D space,
 giving insights into their structure and relationships prior to any transformation by the GCN."""
# Extract feature vectors for t-SNE
X = feature_vectors[:, 1:]  # Exclude node IDs

# Apply t-SNE to reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
for label in range(4):  # Assuming you have 4 classes
    indices = np.where(node_labels[:, 1] == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], label=str(label))
plt.legend()
plt.title('t-SNE visualization of original feature vectors')
plt.show()


"""Takes the preprocessed data and creates tensors for them so they can be used in the GCN model"""
def create_tensors():
    node_ids = feature_vectors[:, 0]
    train_ids, test_ids = train_test_split(node_ids, test_size=0.2, random_state=42)

    train_mask = np.isin(node_ids, train_ids)
    test_mask = np.isin(node_ids, test_ids)

    all_features_tensor = torch.FloatTensor(feature_vectors[:, 1:]).to(device)
    train_labels_tensor = torch.LongTensor(node_labels[train_mask, 1]).to(device)
    test_labels_tensor = torch.LongTensor(node_labels[test_mask, 1]).to(device)
    train_tensor = torch.BoolTensor(train_mask).to(device)
    test_tensor = torch.BoolTensor(test_mask).to(device)
    return all_features_tensor, train_labels_tensor, test_labels_tensor, train_tensor, test_tensor, test_mask
