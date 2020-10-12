#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:05:21 2020

@author: ebukaokpala
"""
from scipy.io import mmread, mminfo
import matplotlib.pyplot as plt
import argparse
import networkx as nx
from collections import Counter
import numpy as np

def construct_graph(matrix):
    """
    Creates a graph from a numpy ndarray
    Arguments:
        matrix: a numpy array of shape m x n
    Returns:
        G (object): A Networkx graph object
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
        None
    """
    
    print("Reading file...")
    sparse_matrix = mmread(file)
    print("File read.")
    print("File info: \n rows : {}, cols: {}, nonzeros: {}, format: {}, field: {}, symmetry: {}".format(*mminfo(file)))
    adjacency_matrix = sparse_matrix.toarray()
    graph = construct_graph(adjacency_matrix)
    return adjacency_matrix, graph


def plot_degree_distribution_histogram(graph):
    """
    Plot the degree distribution histogram of graph and saves the figure
    Arguments:
        graph: A networkx graph object
    Returns:
        None
    """
    #num_of_nodes = len(graph)
    all_degrees = [degree for node, degree in graph.degree()]
    all_degrees = sorted(all_degrees, reverse=True)
    # Count the number of vertices having same degree
    degree_count = Counter(all_degrees)
    degrees, counts = zip(*degree_count.items())
    #degree_distributions = [degree / num_of_nodes for degree in degrees]
    
    plt.hist(degrees, density=True)
    plt.ylabel("Fraction of vertices with degree k")
    plt.xlabel("Degree k")
    plt.savefig("degree_distribution_histogram.png")


def plot_cumm_degree_distribution_function(graph):
    """
    Plot the degree distribution of graph and saves the figure
    Arguments:
        graph: A networkx graph object
    Returns:
        None
    """
    num_of_nodes = len(graph)
    all_degrees = sorted([degree for node, degree in graph.degree()], reverse=True)
    ranks_of_vertices = [i + 1 for i, _ in enumerate(all_degrees)]
    rank_over_num_of_nodes = [rank / num_of_nodes for rank in ranks_of_vertices]
    
    #degree_count = Counter(all_degrees)
    #degrees, counts = zip(*degree_count.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(degrees, counts)
    ax.plot(all_degrees, rank_over_num_of_nodes)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title("Cummulative degree distribution function")
    plt.xlabel("Degree k")
    plt.ylabel("Fraction of vertices pk having degree k or greater")
    fig.savefig("degree_distribution_function.png")
    
    


def compute_power_law_parameters(graph):
    """
    Computes the power law parameters C and alpha
    Arguments:
        graph: networkx graph object
    Returns:
        None
    """
    min_degree = 5
    N = 0
    degrees = []
    for _, degree in graph.degree():
        if degree >= min_degree:
            N += 1
            degrees.append(degree)
    
    sumed = 0
    for degree in degrees:
        value = np.log(degree / (min_degree - 0.5))
        sumed += value
    
    alpha = 1 + N * np.reciprocal(sumed)
    C = (alpha - 1) * min_degree ** alpha - 1
    print("alpha: {}, C: {}".format(alpha, C))
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datalist", nargs="+", type=str, help="input arguments")
    args = parser.parse_args()
    input_array = args.datalist
    assert input_array[0].endswith('.mtx'), "File must be a .mtx file."
    file_name = input_array[0]
    
    _, graph = process_file(file_name)
    plot_degree_distribution_histogram(graph)
    plot_cumm_degree_distribution_function(graph)
    compute_power_law_parameters(graph)
    

if __name__ == "__main__":
    main()