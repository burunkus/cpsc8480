#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:36:00 2020

@author: ebukaokpala
"""

import numpy as np
import numpy.linalg as la
from scipy.io import mmread, mminfo
import argparse
import networkx as nx
from networkx.drawing.nx_pydot import write_dot

def construct_graph(matrix):
    """
    Creates a graph from a numpy ndarray
    Arguments:
        matrix: a numpy array of shape m x n
    Returns:
        G: object
        a Networkx graph object
    """
    
    print("Constructing graph from coordinate file format...")
    G = nx.Graph(matrix)
    print("Construction done.")
    return G


def process_file(file):
    """
    Load a Matrix Market file.
    Arguments:
        file: a matrix market file of type .mtx
    Returns:
        adjacency_matrix: a sparse matrix of shape (number of nodes, number of nodes)
        graph: networkx graph object of the adjacency matrix
    """
    
    print("Reading file...")
    sparse_matrix = mmread(file)
    print("File read.")
    print("File info: rows : {}, cols: {}, nonzeros: {}, format: {}, field: {}, symmetry: {}".format(*mminfo(file)))
    adjacency_matrix = sparse_matrix.toarray()
    graph = construct_graph(adjacency_matrix)
    return adjacency_matrix, graph


def page_rank(adjacency_matrix, graph, damping_factor=0.15, num_iteration=None):
    """
    Creates the transition matrix and uses the power-iteration method to
    calculate the rank.
    inputs:
        adjacency_matrix: a sparse matrix of shape (number of nodes, number of nodes)
        graph: networkx graph object of the adjacency matrix
        damping_factor: constant between 0 and 1 used to mitigate dangling nodes
        num_iteration: the number of time to run the power-iteration method if given.
        if not given, runs power-iteration until convergence.
    Returns:
        rank: numpy array
        a one-dimentional vector of ranks having shape (1, number of nodes) and sums to 1
    """
    
    M = graph.number_of_edges()
    N = graph.number_of_nodes()
    print("Edges: {}, Nodes: {}".format(M, N))
    
    # Handle self links by setting diagonals to zeros
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix = np.transpose(adjacency_matrix)
    
    # Sum of columns
    sum_of_columns = adjacency_matrix.sum(axis=0)
    
    # Create the transition matrix (P)
    print("Creating transition matrix...")
    transition_matrix = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            if adjacency_matrix[i][j] > 0:
                transition_matrix[i][j] = adjacency_matrix[i][j] / sum_of_columns[j]
    print("Creation done.")
    
    # Create a teleportation matrix (T)
    teleportation = np.ones((N, N)) * 1 / N
    # M = dP + (1 - d) / N * T
    markov = damping_factor * transition_matrix + ((1 - damping_factor) / N) * teleportation
    #markov = (1 - damping_factor) * transition_matrix + damping_factor * teleportation
    
    # Create the rank (r) by initializing it to 1/N
    rank = np.empty(N)
    for i in range(0, N):
        rank[i] = 1 / N
    
    if num_iteration:
        for i in range(num_iteration):
            rank = np.multiply(markov, rank)
            print("iteration {}".format(i + 1))
    else:
        # iterate until convergence
        prev_rank = rank
        rank = np.multiply(markov, rank)
        while la.norm(prev_rank - rank) > 0.01:
            prev_rank = rank
            rank = np.multiply(markov, rank)
    
    return rank 


def visualize_graph(graph):
    """
    Save the positions of the nodes to a file. Can be visualized using Gephi
    """
    pos = nx.nx_agraph.graphviz_layout(graph)
    nx.draw(graph, pos=pos)
    write_dot(graph, 'graph.dot')
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datalist", nargs="+", type=str, help="input arguments")
    parser.add_argument("--num", "-i", action="store_true", help="number of iteration flag")
    args = parser.parse_args()
    input_array = args.datalist
    assert input_array[0].endswith('.mtx'), "File must be a .mtx file."
    file_name = input_array[0]
    
    if args.num:
        num_iteration = int(input_array[1])
    else:
        num_iteration = None
        
    # Create an adjacency matrix (A)
    adjacency_matrix, graph = process_file(file_name)
    rank = page_rank(adjacency_matrix, graph, num_iteration=num_iteration)
    print(rank)
    #visualize_graph(graph)
    
    
if __name__ == "__main__":
    main()