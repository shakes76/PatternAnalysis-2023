import csv
import numpy as np

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

adjacency_matrix = create_adjacency_matrix()
print(adjacency_matrix)