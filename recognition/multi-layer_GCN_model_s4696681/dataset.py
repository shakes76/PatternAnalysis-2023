import csv
import numpy as np
from scipy.linalg import sqrtm
import json

"""Create Adjacency matrix from the musae_facebook_edges.csv file"""
def create_adjacency_matrix():
    data = []

    # Read the CSV data from musae_facebook_edges.csv file
    with open('../../../facebook_large/musae_facebook_edges.csv', 'r') as file:
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

#adjacency_normed = normalise_adjacency_matrix(create_adjacency_matrix())
#print(adjacency_normed)


"""Since the feature .json file has an inconsistent number of features for each node.
 We need to make them consistent for training.
 Hence I will create an n-dimensional bag of words feature vector for each node"""
def create_feature_vectors():
    # Load data from a JSON file
    with open('../../../facebook_large\musae_facebook_features.json', 'r') as file:
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
print(feature_vectors)

