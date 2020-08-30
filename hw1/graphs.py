#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 22:56:07 2020

@author: ebukaokpala
"""

import numpy as np
from scipy.io import mmread, mminfo
from collections import deque
import argparse
import networkx as nx

def breath_first_search(graph, source):
    """
    Runs Breath First Search on a graph starting from a source.
    Arguments:
        graph: a networkx graph object
        source (int): the starting node of breath first search traversal.
    Returns:
        None
    """
    
    print("Running Breath First Search...")
    queue = deque([])
    visited = set()
    root = source
    queue.append(root)
    visited.add(root)
    path = []
    
    while queue:
        root = queue.popleft()
        path.append(root)
        for node in graph.neighbors(root): #or for node in graph[root]: pass
            if node not in visited:
                visited.add(node)
                queue.append(node)
                
    print("Breath First Search complete.")
    print("Comparing bfs with networkx bfs implementation...")
    edges = nx.bfs_edges(graph, source)
    nodes = [source] + [v for u, v in edges]
    if path == nodes:
        print("Both implementation are same.")
    else:
        print("They are not the same.")


def add_attributes(graph, node=None, edge=(None, None), node_attr=None,\
                   node_attr_val=None, edge_attr=None, edge_attr_val=None):
    """
    Adds a new node or edge with attributes to a graph. 
    Arguments:
        graph: a networkx graph object
        node: a node to add to the graph. Must be a hashable object
        edge: a tuple containing a pair of node (u, v) to add to the graph.
        node_attr: the name of the node attribute.
        node_attr_val: the value of the node attribute. 
        edge_attr: the name of the edge attribute.
        edge_attr_val: the value of the edge attribute
    Returns:
        None
    """
    
    if node:
        graph.add_node(node)
        graph.nodes[node][node_attr] = node_attr_val
        print("Node attribute added.")
       
    if edge != (None, None):
        node1, node2 = edge
        if edge_attr == "weight":
            graph.add_edge(node1, node2)
            graph[node1][node2][edge_attr] = float(edge_attr_val)
           
        else:
            graph.add_edge(node1, node2)
            graph[node1][node2][edge_attr] = edge_attr_val
        print("Edge attribute added.")


def view_node_attributes(graph, name):
    print(nx.get_node_attributes(graph, name))


def view_edge_attribute(graph, u, v, name):
    all_attributes = nx.get_edge_attributes(graph, name)
    print("The edge attribute value added: ", all_attributes[(u, v)])


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


def eigenvalues_of_graph_laplacian(graph, num_eigenvalues=5):
    """
    Calculates the smallest eigenvalues of the Laplacian of a graph adjacency matrix.
    Arguments:
        graph: networkx graph object
        num_eigenvalues: number of eignenvalues to compute, defaults to 5. int
    Return:
        None
    """
    
    print("Computing {} smallest eigenvalues of graph laplacian...".format(num_eigenvalues))
    laplacian_matrix = nx.laplacian_matrix(graph)
    eigenvalues = np.linalg.eigvals(laplacian_matrix.A)
    eigenvalues = np.sort(eigenvalues)
    smallest = [eigenvalues[i] for i in range(num_eigenvalues)]
    print("Smallest eigenvalues: {} \n".format(smallest))


def visualize_graph(graph):
    print("Drawing graph using Graphviz...")
    A = nx.nx_agraph.to_agraph(graph)
    A.layout()
    print("Saving graph as an image...")
    A.draw("graph.png")
    print("Wrote graph.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datalist", nargs="+", type=str, help="input arguments")
    parser.add_argument("--node", "-n", action="store_true", help="node attribute addition flag")
    parser.add_argument("--edge", "-e", action="store_true", help="edge attribute addition flag")
    parser.add_argument("--viewnodes", "-vn", action="store_true", help="view node attributes flag")
    parser.add_argument("--viewedge", "-ve", action="store_true", help="view edge attributes flag")
    args = parser.parse_args()
    input_array = args.datalist
    assert input_array[0].endswith('.mtx'), "File must be a .mtx file."
    file_name = input_array[0]
    
    if args.node:
        assert len(input_array[1:]) >= 3, "Node, attribute name and attribute value must be specified."
        node, node_attr, node_attr_val = input_array[1], input_array[2], input_array[3]
    if args.edge:
        assert len(input_array[1:]) >= 4, "Edge(i.e u and v), attribute name and attribute value must be specified."
        u, v, edge_attr, edge_attr_val = input_array[1], input_array[2], input_array[3], input_array[4]
    if args.node and args.edge:
        assert len(input_array[1:]) >= 7, "Node, node attribute name, node attribute value, \
        edges u and v, edge attribute name, edge attribute value must be given."
        node, node_attr, node_attr_val = input_array[1], input_array[2], input_array[3]
        u, v, edge_attr, edge_attr_val = input_array[5], input_array[6], input_array[7], input_array[8]
    if args.node and args.viewnodes:
        assert len(input_array[1:]) >= 4, "The node attribute name to view must be specified"
        node_name = input_array[4]
    if args.edge and args.viewedge:
        assert len(input_array[1:]) >= 5, "The edge attribute name to view must be the specified"
        edge_name = input_array[5]
        
    if args.node and args.edge and args.viewnodes and args.viewedge:
        assert len(input_array[1:]) >= 9, "Node, node attr, node attr value, edge(u, v),\
        edge attr, edge attr value, node attr name to view and edge attr name to view\
        must be specified."
        node_name, edge_name = input_array[4], input_array[9]
        
    
    adjacency_matrix, graph = process_file(file_name)
    visualize_graph(graph)
    breath_first_search(graph, list(graph.nodes)[0])
    eigenvalues_of_graph_laplacian(graph)
    
    if args.node:
        add_attributes(graph, node=node, node_attr=node_attr, node_attr_val=node_attr_val)
    if args.edge:
        add_attributes(graph, edge=(u, v), edge_attr=edge_attr, edge_attr_val=edge_attr_val)
    
    if args.viewnodes:
        view_node_attributes(graph, node_name)
    if args.viewedge:
        view_edge_attribute(graph, u, v, edge_name)

if __name__ == "__main__":
    main()