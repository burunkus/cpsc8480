import matplotlib.pyplot as plt
import argparse
import networkx as nx
import numpy as np
import time
from scipy import stats
from pprint import pprint
import os


def create_cocitation_network(citation_net, file_name):
    """
    creates a cocitation network where two papers are cocited (having an edge between them)
    if they are both cited by a third paper. This cocitation network is written as edge list 
    to a file. 
    Arguments:
        citation_net: A Networkx directed graph object
    Returns:
        None
    """
    cocited = set()
    with open(file_name, 'w') as file:
        for node, neighbors in citation_net.adj.items():
            cites = list(citation_net.neighbors(node))
            size = len(cites)
            for i in range(size):
                for j in range(i, size):
                    if i == j: continue
                    edge = (cites[i], cites[j])
                    if edge not in cocited:
                        cocited.add(edge)
                        file.write(f"{edge[0]} {edge[1]} \n")
                    

def create_network_by_year(nodes_file, dates_file):
    """
    Creates a file of directed graph nodes where a paper i cites paper j within 
    a specific year range. counts the number of papers published in each year and writes them
    to a file.
    Arguments:
        nodes_file: a .txt file containing FromNodeId and ToNodeId in a directed network
        dates_file: a .txt file containing NodeId and the date of its publication
    Returns:
        None
    """
    years = [{1992, 1993, 1994}, {1992, 1993, 1994, 1995, 1996},
             {1992, 1993, 1994, 1995, 1996, 1997, 1998},
             {1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000},
             {1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002}]
    
    seen = set()
    file_names = ["92-94", "92-96", "92-98", "92-00", "92-02"]
    for i, years_set in enumerate(years):
        seen.clear()
        file_object = open(file_names[i] + ".txt", 'a')
        with open(dates_file) as file:
            for i, line in enumerate(file):
                if i == 0: continue
                nodeid, date = line.split()
                if nodeid[0:2] == "11":
                    nodeid = nodeid[2:]
                if int(date.split('-')[0]) in years_set:
                    with open(nodes_file) as f:
                        for i, row in enumerate(f):
                            if i == 0 or i == 1 or i == 2 or i == 3: continue
                            from_node_id, to_node_id = row.split()
                            if from_node_id == nodeid:
                                if row not in seen:
                                    file_object.write(row)
                                    seen.add(row)
        file_object.close()
    
    paper_count = {}
    with open(dates_file) as file:
        for i, line in enumerate(file):
            if i == 0: continue
            _, date = line.split()
            year = int(date.split('-')[0])
            if year in paper_count:
                paper_count[year] += 1
            else:
                paper_count[year] = 1
    
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    with open(path + "paper_count_per_year.txt", 'a') as file:
        for year, count in paper_count.items():
            file.write(f"{year} {count} \n")
            

def get_components(citation_net, is_directed=False):
    """
    Computes the weakly connected components of citation_net
    Arguments:
        citation_net: A Networkx undirected/directed graph object
        is_directed: Boolean:
                     If True, citation_net is assumed to be directed, otherwise,
                     it is assumed to be undirected. Defaults to False
    Returns:
        number_of_cc: The number of connected components in citation_net
        largest_cc: The largest connected component
        subgraph_largest_cc: The subgraph induced by the largest_cc
    """
    if is_directed:
        largest_cc = max(nx.weakly_connected_components(citation_net), key=len)
        number_of_cc = nx.number_weakly_connected_components(citation_net)
        subgraph_largest_cc = citation_net.subgraph(largest_cc).copy()
    else:
        largest_cc = max(nx.connected_components(citation_net), key=len)
        number_of_cc = nx.number_connected_components(citation_net)
        subgraph_largest_cc = citation_net.subgraph(largest_cc).copy()
        
    return number_of_cc, largest_cc, subgraph_largest_cc


def get_mean_citation_per_paper(citation_net, is_directed=False):
    """
    Computes the mean citation per paper
    Arguments:
        citation_net: A NetworkX graph object
        is_directed: Boolean: If True, gets the number of out degrees of a node 
        in a directed graph, otherwise, it gets the number of edges incident to a node
        in an undirected graph. Defaults to False.
    Returns:
        mean_citation_per_paper: float: the mean citation per paper in the graph
    """
    N = citation_net.number_of_nodes()
    sum_of_citations = 0
    if is_directed:
        for node, neighbors in citation_net.adj.items():
            sum_of_citations += citation_net.in_degree(node)
    else:
        for node, neighbors in citation_net.adj.items():
            sum_of_citations += citation_net.degree(node)
    mean_citation_per_paper = sum_of_citations / N
    return mean_citation_per_paper


def get_summary_statistics(citation_net, A, is_directed=False):
    """
    Generates the overall statistics (number of components, largest component,
    mean distance, mean citations per paper, number of papers that cite each other and the mean) 
    of a directed graph. 
    Arguments:
        citation_net: A NetworkX directed graph object
        A: A numpy array of shape n x n containing the adjacency matrix of citation_net
    Returns:
        None
    """
    N = citation_net.number_of_nodes()
    E = citation_net.number_of_edges()
    
    # Components
    number_of_cc, largest_cc, subgraph_largest_cc = get_components(citation_net, is_directed=True)
    N_SUB = subgraph_largest_cc.number_of_nodes()
    
    # Mean distance
    start_time = time.time()
    mean_distance = nx.average_shortest_path_length(subgraph_largest_cc)
    
    # Citations per paper
    mean_citation_per_paper = get_mean_citation_per_paper(citation_net, is_directed=True)
   
    #Assortative mixing
    largest_wcc_assortativity = assortative_mixing(subgraph_largest_cc)
    assortativity_whole_net = assortative_mixing(citation_net)
    
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    with open(path + "main_overall_stats", 'w') as f:
        f.write(f"Number of nodes: {N}, number of edges: {E} \n")
        f.write(f"Size of largest connected components: {len(largest_cc)} \n")
        f.write(f"There are {N_SUB} papers in this subgraph \n")
        f.write(f"Mean distance in the largest component: {mean_distance} \n")
        f.write(f"The mean citation per paper: {mean_citation_per_paper} \n")
        f.write(f"Assortative mixing in the largest weakly connected component: {largest_wcc_assortativity} \n")
        f.write(f"Assortative mixing in the whole network: {assortativity_whole_net} \n")


def get_summary_statistics_by_year(file_names):
    """
    Computes the statistics (number of paper, number of papers in the largest component,
    the ratio, mean distance and the number of papers that cite each other) of the papers
    published in year ranges in file_names and writes that statistics to a file. 
    Arguments:
        file_names: a list where each element is a text file containing FromNodeId ToNodeId of 
        paper citations in the year range specified by the file name. 
    Returns:
        None
    """
    for file in file_names:
        citation_net = nx.read_edgelist(file, nodetype=int)
        N = citation_net.number_of_nodes()
        _, largest_cc, subgraph_largest_cc = get_components(citation_net, is_directed=False)
        size_of_largest_cc = len(largest_cc)
        number_of_papers_in_subgraph = subgraph_largest_cc.number_of_nodes()
        ratio = size_of_largest_cc / N
        mean_distance = nx.average_shortest_path_length(subgraph_largest_cc)
        mean_citation_per_paper = get_mean_citation_per_paper(citation_net)
        #Assortative mixing
        largest_wcc_assortativity = assortative_mixing(subgraph_largest_cc)
        assortativity_whole_net = assortative_mixing(citation_net)
        
        path = "cocitation-results/"
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + "summary_result_for_" + file, 'w') as f:
            f.write(f"Number of papers is: {N} \n")
            f.write(f"The mean citation per paper: {mean_citation_per_paper} \n")
            f.write(f"Size of the largest connected component is: {size_of_largest_cc} \n")
            f.write(f"The ratio is: {ratio} \n")
            f.write(f"The mean distance is: {mean_distance} \n")
            f.write(f"Assortative mixing in the largest weakly connected component: {largest_wcc_assortativity} \n")
            f.write(f"Assortative mixing in the whole network: {assortativity_whole_net} \n")


def get_cocitation_summary_statistics(cocitation_net):
    """
    Generates the overall statistics (number of components, largest component,
    mean distance) of an undirected graph. 
    Arguments:
        cocitation_net: A NetworkX undirected graph object
    Returns:
        None
    """
    N = cocitation_net.number_of_nodes()
    E = cocitation_net.number_of_edges()
    
    # Components
    number_of_cc, largest_cc, subgraph_largest_cc = get_components(cocitation_net)
    N_SUB = subgraph_largest_cc.number_of_nodes()
    size_of_largest_cc = len(largest_cc)
    
    # Mean distance
    mean_distance = nx.average_shortest_path_length(subgraph_largest_cc)
   
    #Assortative mixing
    largest_cc_assortativity = assortative_mixing(subgraph_largest_cc)
    assortativity_whole_net = assortative_mixing(cocitation_net)
    ratio = size_of_largest_cc / N
    
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    with open(path + "cocitation_main_overall_stats", 'w') as f:
        f.write(f"Number of nodes: {N}, number of edges: {E} \n")
        f.write(f"Size of largest connected components: {size_of_largest_cc} \n")
        f.write(f"The ratio is: {ratio} \n")
        f.write(f"There are {N_SUB} papers in this subgraph \n")
        f.write(f"Mean distance in the largest component: {mean_distance} \n")
        f.write(f"Assortative mixing in the largest weakly connected component: {largest_cc_assortativity} \n")
        f.write(f"Assortative mixing in the whole network: {assortativity_whole_net} \n")
        

def get_cocitation_summary_statistics_by_year(file_names):
    """
    Computes the statistics (number of paper, number of papers in the largest component,
    the ratio, mean distance and the number of papers that cite each other) of the papers
    published in year ranges in file_names and writes that statistics to a file. 
    Arguments:
        file_names: a list where each element is a text file of cocitation network containing two nodes(papers) that are
        connected by an edge. 
    Returns:
        None
    """
    for file in file_names:
        cocitation_net = nx.read_edgelist(file, nodetype=int)
        N = cocitation_net.number_of_nodes()
        E = cocitation_net.number_of_edges()
        number_of_cc, largest_cc, subgraph_largest_cc = get_components(cocitation_net)
        size_of_largest_cc = len(largest_cc)
        number_of_papers_in_subgraph = subgraph_largest_cc.number_of_nodes()
        ratio = size_of_largest_cc / N
        #mean_distance = nx.average_shortest_path_length(subgraph_largest_cc)
        #Assortative mixing
        largest_cc_assortativity = assortative_mixing(subgraph_largest_cc)
        assortativity_whole_net = assortative_mixing(cocitation_net)
        
        path = "cocitation-results/"
        if not os.path.exists(path):
            os.mkdir(path)
            
        with open(path + "summary_result_for_" + file, 'w') as f:
            f.write(f"Number of papers is: {N}, number of edges: {E} \n")
            f.write(f"Size of the largest connected component is: {size_of_largest_cc} \n")
            f.write(f"The ratio is: {ratio} \n")
            f.write(f"Assortative mixing in the largest weakly connected component: {largest_cc_assortativity} \n")
            f.write(f"Assortative mixing in the whole network: {assortativity_whole_net} \n")
            
            
            
def get_centralities(citation_net, is_directed=False):
    """
    Computes the centralities(degree, betweenness, closeness and page rank) of the 
    undirected cocitation graph citation_net
    Arguments:
        citation_net: A NetworkX undirected graph object
    Returns:
        degree_cen: (dict) - dictionary of nodes having degree centrality as value
        betweenness_cen: (dict) - dictionary of nodes having betweenness centrality as value
        closeness_cen: (dict) - dictionary of nodes having closeness centrality as value
        page_rank: (dict) - dictionary of nodes having rank as value
    """
    #_, largest_cc, _ = get_components(citation_net, is_directed=False)
    #subgraph_largest_cc = citation_net.subgraph(largest_cc).copy()
    degree_cen = nx.degree_centrality(citation_net)
    page_rank = nx.pagerank(citation_net)
    betweenness_cen = nx.betweenness_centrality(citation_net)
    closeness_cen = nx.closeness_centrality(citation_net)
    return degree_cen, betweenness_cen, closeness_cen, page_rank       
        

def frequency_distribution(citation_net, file_names, centralities=None, in_degree=False, is_directed=False):
    """
    Plot the degree distribution histogram of graph and saves the figure
    Arguments:
        citation_net: A networkx graph object
        file_names: list: where each element is a unique name to save the histogram as
        centralities: list: where each element is a centrality measure dictionary
        having nodeid as key and value(rank) of the nodid as value. Default to None.
        in_degree: Boolean: If True, plots the In-degree of citation_net, otherwise, plots
        the Out-degree. Only if is_directed is True, see is_directed below.
        is_directed: Boolean: If True, plots distribution for a directed network, otherwise, 
        plots for an undirected network. Defaults to False. 
    Returns:
        None
    learn-more: http://www.mkivela.com/binning_tutorial.html
    """
    n_bins = 40
    if centralities:
        fig = plt.figure(figsize=(13,13))
        for i, centrality in enumerate(centralities):
            centrality = list(centrality.values())
            ax = fig.add_subplot(2, 2, i+1)
            ax.hist(centrality, n_bins, density=True) #lin-lin scale
            plt.ylabel("Frequency")
            plt.xlabel(file_names[i])
        plt.savefig("distribution_histogram_lin_lin_scale.png")
        
        fig = plt.figure(figsize=(13,13))
        for i, centrality in enumerate(centralities):
            centrality = list(centrality.values())
            ax = fig.add_subplot(2, 2, i+1)
            ax.hist(centrality, n_bins, density=True)
            ax.set_yscale('log', nonposy='clip') #lin-log scale
            plt.ylabel("Frequency")
            plt.xlabel(file_names[i])
        plt.savefig("distribution_histogram_lin_log_scale.png")
            
    else:
        if is_directed:
            if in_degree:
                degrees = [degree for _, degree in citation_net.in_degree()]
            else:
                degrees = [degree for _, degree in citation_net.out_degree()]
        else:
            degrees = [degree for _, degree in citation_net.degree()]
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.hist(degrees, n_bins, density=True)
        ax.set_yscale('log', nonposy='clip') #lin-log scale
        plt.ylabel("Frequency")
        plt.xlabel(file_names[0])
        plt.savefig(file_names[0] + "_distribution_histogram.png")


def plot_cummulative_centrality_distribution(citation_net, file_names, centralities):
    """
    plots the cummulative distribution function of a list of centrality measures.
    Argument:
        citation_net: A NetworkX directed/undirected graph object
        centralities: List: each element is a list of a centrality measure dictionary - nodeid: value/ranks
        file_names: List: each element is the name to be given to each centrality in centralities
        upon saving the plot to your current working directory. 
    Returns:
        None
    """

    num_of_nodes = citation_net.number_of_nodes()
    for i, centrality in enumerate(centralities):
        centrality = sorted(centrality.values(), reverse=True)
        ranks_of_vertices = [i + 1 for i, _ in enumerate(centrality)]
        rank_over_num_of_nodes = [rank / num_of_nodes for rank in ranks_of_vertices]
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(centrality, rank_over_num_of_nodes)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i != 3:
            plt.xlabel(file_names[i] + " centrality x")
            plt.ylabel("Fraction of vertices having centrality x or greater")
        else:
            plt.xlabel(file_names[i])
            plt.ylabel("Fraction of vertices having rank x or greater")
        fig.savefig(file_names[i] + "_cummulative_distribution_function.png")
        plt.show()
    
    
def plot_papers_vs_years(file):
    """
    Plots the number of papers published each year. 
    Arguments:
        file: .txt file
              The name of the file containing on each row, the year and number of papers
              published that year.
    Returns:
        None
    """
    years, counts = [], []
    with open(file) as f:
        for line in f:
            year, count = line.split()
            years.append(int(year))
            counts.append(int(count))
    years, counts = np.asarray(years), np.asarray(counts)
    fig, ax = plt.subplots()
    ax.plot(years, counts)
    ax.set(xlabel='Year', ylabel='Number of papers')
    ax.grid()
    fig.savefig("papers_vs_counts.png")
    plt.show()
    
    
def assortative_mixing(citation_net):
    """
    Computes the correlation coefficient r to quantify assortative mixing in 
    citation_net.
    Arguments:
        citation_net: A NetworkX graph object. Will be converted to undirected 
        if directed.
    Returns:
        r: float: the correlation coefficient. Implementation is based on equation(8.27)
        of Newman's text - Networks an Introduction.
    """
    if citation_net.is_directed():
        citation_net = citation_net.to_undirected()
        
    edges = set([(u, v) for u, v in citation_net.edges() if u != v])
    degree_sum = 0
    for u, v in edges:
        degree_sum += citation_net.degree(u) * citation_net.degree(v)
    sc = 2 * degree_sum
    s1 = sum(citation_net.degree(i) for i in citation_net.nodes)
    s2 = sum(citation_net.degree(i) ** 2 for i in citation_net.nodes)
    s3 = sum(citation_net.degree(i) ** 3 for i in citation_net.nodes)
    r = ((s1 * sc) - (s2 ** 2)) / ((s1 * s3) - s2 ** 2)
    return r

def get_citation_count(citation_net):
    """
    Computes citation count (the total number) of papers that cited a paper in citation_net.
    Arguments:
        citation_net: A NetworkX directed graph object
    Returns:
        citation_count: dictionary
                        contains the nodeid(paper) as key and total number of citations it received
                        as value. 
    """
    
    citation_count = {}
    for node, neigbors in citation_net.adj.items():
        if node in citation_count:
            citation_count[node] += citation_net.in_degree(node)
        else:
            citation_count[node] = citation_net.in_degree(node)
    return citation_count
        
        
def get_top_values(dictionary, top_k):
    """
    Returns the top k elements in dictionary based on dictionary values.
    Argument:
        dictionary: A dictionary: keys are nodes and values are the values(rankings) of different centrality measures
        top_k: Integer: a threshold parameter
    Returns:
        An iterator of tuples(nodeid, value) of the top_k values in dictionary
    """
    items = sorted(dictionary.items(), key=lambda x : x[1], reverse=True)
    return map(lambda x : (x[0], x[1]), items[:top_k])        
        

def get_spearman_correlation(array1, array2, names, citation_count_ranking, centrality_rankings):
    """
    Computes the Spearmans correlation between each of the elements in 
    array1 and array2. Plots and save the scatter plot between citation count and each
    centrality measure rankings.
    Arguments:
        array1: List[List]: multi-dimensional array
                Each element is a list of a centrality measure values execept for the list
                at index 0, which is a list of citation count.
        array2: List[List]: multi-dimensional array
                Copy of array1
        names:  List: 
                List of names given to each computation for easy identification
        citation_count_ranking: dict:
                                A dictionary of the citation count ranking of the directed citation network having key as nodeid and value as rank
        centrality_rankings: List:
                             Each element is a dict of a centrality measure ranking, the key is a nodeid and value is its rank.
    Returns:
        None
    """
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    with open(path + "correlations" + ".txt", 'w') as f:
        for i, data in enumerate(array1):
            for j, data2 in enumerate(array2):
                rho, pval = stats.spearmanr(data, data2)
                f.write(f"correlation of {names[i], names[j]}: rho: {rho}, pval: {pval} \n")
                
    file_names = names[1:]
    fig = plt.figure(figsize=(13,13))
    for i, centrality in enumerate(centrality_rankings):
        # Ensure that the citation ranks and centrality ranks are vectors of the same length, since some nodes
        # in citation_count_ranking might not be present in centrality.
        citation_ranks, centrality_ranks = [], []
        for nodeid, rank in citation_count_ranking.items():
            if nodeid in centrality:
                citation_ranks.append(rank)
                centrality_ranks.append(centrality[nodeid])
                
        ax = fig.add_subplot(2, 2, i+1)
        ax.scatter(sorted(citation_ranks), sorted(centrality_ranks, reverse=True))
        plt.ylabel(file_names[i] + " centrality rankings")
        plt.xlabel("Citation rankings")
    plt.savefig("centrality_ranking_vs_citation_scatter.png")
    

def interpret(centralities, file_names, year, ranking=False):
    """
    Writes to a file, the top nodeid's (papers) based on centrality measures
    Arguments:
        centralities: List: 
                      Each element is an iterator over tuples of top papers and their values
        file_names: List:
                    Each element is the name (String) of the file to save the nodeid and value calculated by
                    centrality measures whose results are contained in centralities.
        year: String:
              The year the centrality measure was computed in, used to ensure uniqueness of file names
              if file_names
        ranking: Boolean:
                 If True, the ranking of nodeids will be saved, otherwise, it won't. Defaults to False.
    Returns:
        None
    """
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
    for i, centrality in enumerate(centralities):
        with open(path + file_names[i] + "_" + year + ".txt", 'w') as file:
            if ranking:
                for nodeid, rank in centrality:
                    file.write(f"{nodeid} {rank}\n")
            else:
                for nodeid, _ in centrality:
                    file.write(f"{nodeid} \n")
                
                
def rank_citation_count(citation_count, file_name):
    """
    Writes to a file, the nodeid's (papers) ranked by the number the citations.
    Arguments:
        citation_count: dict:
                        dictionary of key(nodeid), value(citation count) pairs
        file_name: String:
                   The name of the file to save data to
    Returns:
        rankings: dict: A dictionary having nodeid as key and the rank of the nodeid as value
    """
    rankings = {}
    prev = ""
    rank = 0
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + file_name + ".txt", 'w') as file:
        for nodeid, value in sorted(citation_count.items(), key=lambda x : x[1], reverse=True):
            if value != prev:
                rank += 1
                file.write(f"{nodeid} {value} {rank} \n")
                prev = value
                rankings[nodeid] = rank
            else:
                file.write(f"{nodeid} {value} {rank} \n")
                prev = value
                rankings[nodeid] = rank
    return rankings


def rank_centralities(centralities, file_names):
    """
    Computes the ranks of the keys in the centrality measures in centralities 
    based on the values. 
    Arguments:
        centralities: List:
                      Each element is a dict of a centrality measure where the key is a 
                      node(paper) and the value is the value computed by the centrality measure
        file_names: List:
                    Each element is a name to save the rankings of each centrality measure as.
    Returns:
        rankings: List:
                  Each element is a dict for each centrality measure in centralities,
                  the key of the dict is a nodeid and the value is the rank.
    """
    rankings = []
    prev = ""
    rank = 0
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
    for i, centrality in enumerate(centralities):
        rank = 0
        ranking = {}
        with open(path + file_names[i] + ".txt", 'w') as file:
            for nodeid, value in sorted(centrality.items(), key=lambda x : x[1], reverse=True):
                if value != prev:
                    rank += 1
                    file.write(f"{nodeid} {value} {rank} \n")
                    prev = value
                    ranking[nodeid] = rank
                else:
                    file.write(f"{nodeid} {value} {rank} \n")
                    prev = value
                    ranking[nodeid] = rank
        rankings.append(ranking)
    return rankings

                    
def save(centralities, citation_count, file_names):
    """
    Saves the nodeid, citation count and centrality value to different files based on citation count. 
    Arguments:
        centralities: List: 
                      Each element is a dictionary of a centrality measure, where the key is a nodeid(paper) 
                      and the value is the calculated centrality value/rank.
        citation_count: dict:
                        The key is a nodeid and value is the citation count.
        file_names: List:
                    Each element is the name (String) of the file to save the nodeid, citation count and centrality value.
    Returns:
        None
    """
    for i, centrality in enumerate(centralities):
        with open(file_names[i] + ".txt", "w") as file:
            """
            file.write(f"#Nodeid  #citation count  #centrality value \n")
            for nodeid, value in sorted(citation_count.items(), key=lambda x : x[1], reverse=True):
                if nodeid in centrality:
                    file.write(f"{nodeid} {citation_count[nodeid]} {centrality[nodeid]} \n")
            """
            file.write(f"#Nodeid #centrality value \n")
            for nodeid, value in centrality.items():
                file.write(f"{nodeid} {value} \n")

                
def read(file_names):
    """
    Reads each file in file_names and creates a vector of centrality values.
    Arguments:
        file_names: List: Each element is a file name
    Returns:
        degree_cen: List: A vector of degree centrality values
        betweenness_cen: List: A vector of betweenness centrality values
        closeness_cen: List: A vector of closeness centrality values
        page_rank: List: A vector of page rank values
    """
    citation_counts, degree_cen, betweenness_cen, closeness_cen, page_rank = [], [], [], [], []
    for file in file_names:
        with open(file + ".txt") as f:
            for i, line in enumerate(f):
                if i == 0: continue
                node, citation_count, centrality_value = line.split()
                citation_counts.append(float(citation_count))
                if file == "degree_cen":
                    degree_cen.append(float(centrality_value))
                elif file == "betweenness_cen":
                    betweenness_cen.append(float(centrality_value))
                elif file == "closeness_cen":
                    closeness_cen.append(float(centrality_value))
                else:
                    page_rank.append(float(centrality_value))
                    
    return citation_counts, degree_cen, betweenness_cen, closeness_cen, page_rank


def small_world_analysis(cocitation_net):
    number_of_cc, largest_cc, subgraph_largest_cc = get_components(cocitation_net)
    number_of_papers = subgraph_largest_cc.number_of_nodes()
    number_of_edges = subgraph_largest_cc.number_of_edges()
    cluster_coefficient = nx.clustering(subgraph_largest_cc)
    cluster_coefficient = sum(list(cluster_coefficient.values())) / number_of_papers
    path_length = nx.average_shortest_path_length(subgraph_largest_cc)
    
    # Random graph of the same details
    random_graph = nx.fast_gnp_random_graph(number_of_papers, 0.75, seed=None, directed=False)
    random_number_of_cc, random_largest_cc, random_subgraph_largest_cc = get_components(random_graph)
    N = nx.number_of_nodes(random_subgraph_largest_cc)
    random_cluster_coefficient = nx.clustering(random_subgraph_largest_cc)
    random_cluster_coefficient = sum(list(random_cluster_coefficient.values())) / N
    random_path_length = nx.average_shortest_path_length(random_subgraph_largest_cc)
    
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + "small-world-analysis.txt", 'w') as f:
        f.write(f"Number of papers is in the cocitation network: {number_of_papers}, number of edges: {number_of_edges} \n")
        f.write(f"Clustering coefficient in the cocitation network: {cluster_coefficient}\n")
        f.write(f"path length in the cocitation network: {path_length}\n")
        f.write(f"Clustering coefficient in the random network: {random_cluster_coefficient}\n")
        f.write(f"path length in the random network: {random_path_length}")
        
    
def main():
    
        # Get the general statistics for the whole directed citation network
    file_name = "cit-HepPh.txt"
    citation_net = nx.read_edgelist(file_name, create_using=nx.DiGraph(), nodetype=int)
    A = nx.to_numpy_array(citation_net, nodelist=sorted(citation_net.nodes))
    print("Getting the general statistics of the directed citation network.")
    get_summary_statistics(citation_net, A)
    
    # Assortative mixing of the citation network
    r = nx.degree_assortativity_coefficient(citation_net)
    print(f"Assortative coefficient for the citation network is: {r}\n")
    
    #Save the citation count for the whole directed citation network
    print("Saving the citaiton count of the directed citation network.")
    citation_count = get_citation_count(citation_net)
    citation_count_ranking = rank_citation_count(citation_count, "citation_count")
    
    # Create different networks according to the date the papers were published
    nodes_file = "cit-HepPh.txt"
    dates_file = "cit-HepPh-dates.txt"
    print("Creating citation networks using the dates file.")
    create_network_by_year(nodes_file, dates_file)
    
    # Get the general statistics for the directed citation network by year
    file_names = ["92-94.txt", "92-96.txt", "92-98.txt", "92-00.txt", "92-02.txt"]
    print("Processing the general statistics of the citation networks by year.")
    get_summary_statistics_by_year(file_names)
    
    # Create cocitation networks by year
    cocitation_file_names = ["cocitation-92-94.txt", "cocitation-92-96.txt", "cocitation-92-98.txt", "cocitation-92-00.txt", "cocitation-92-02.txt"]
    file_names = ["92-94.txt", "92-96.txt", "92-98.txt", "92-00.txt", "92-02.txt"]
    print("Creating cocitation networks by year.")
    for i, file in enumerate(file_names):
        net = nx.read_edgelist(file, nodetype=int)
        create_cocitation_network(net, cocitation_file_names[i])
    
    # Get the general statistics for the yearly undirected cocitation network
    print("Processing the general statistics of the yealy cocitation networks")
    get_cocitation_summary_statistics_by_year(cocitation_file_names)
    
    # Assortative mixing for each of the evolving undirected cocitation network
    print("calculating assortative mixing for the cocitation networks.")
    for net in cocitation_file_names:
        cocitation_net = nx.read_edgelist(net, nodetype=int)
        r = nx.degree_assortativity_coefficient(cocitation_net)
        print(f"Assortative coefficient for {net} is: {r}\n")
    
    # Compute centralities of the yearly cocitation networks
    print("Computing centralities of the yearly cocitation networks")
    for year in cocitation_file_names:
        print(f"Computing centralities for: {year}")
        cocitation_net = nx.read_edgelist(year, nodetype=int)
        _degree_cen, _betweenness_cen, _closeness_cen, _page_rank = get_centralities(cocitation_net)
    
        # Get top 30 papers
        top_k = 30
        top_degree_cen = get_top_values(_degree_cen, top_k)
        top_betweenness_cen = get_top_values(_betweenness_cen, top_k)
        top_closeness_cen = get_top_values(_closeness_cen, top_k)
        top_page_rank = get_top_values(_page_rank, top_k)

        # interpret
        file_names = ["top_degree_cen", "top_betweenness_cen", "top_closeness_cen", "top_page_rank"]
        year = year.split('.')[0]
        interpret([top_degree_cen, top_betweenness_cen, top_closeness_cen, top_page_rank], file_names, year)
        file_names = ["all_degree_cen_cocitation-92-02", "all_betweenness_cen_cocitation-92-02", "all_closeness_cen_cocitation-92-02", "all_page_rank_cocitation-92-02"]
        rankings = rank_centralities([_degree_cen, _betweenness_cen, _closeness_cen, _page_rank], file_names)
    
    
    # Plot frequency and cummulative distributions for the 92-02 undirected cocitation network
    print("Ploting frequency and cummulative distributions.")
    files = ['all_degree_cen_cocitation-92-02.txt', 'all_betweenness_cen_cocitation-92-02.txt',
    'all_closeness_cen_cocitation-92-02.txt', 'all_page_rank_cocitation-92-02.txt']
    path = "cocitation-results/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    file_names = ["Degree", "Betweenness", "Closeness", "Page Rank"]
    centralities = []
    for file in files:
        centrality_dict = {}
        with open(path + file, 'r') as f:
            for line in f:
                nodeid, value, _ = line.split()
                centrality_dict[nodeid] = float(value)
        centralities.append(centrality_dict)
    
    file = "cocitation-92-02.txt"
    cocitation_net = nx.read_edgelist(file, nodetype=int)
    frequency_distribution(cocitation_net, file_names, centralities=centralities)
    plot_cummulative_centrality_distribution(cocitation_net, file_names, centralities=centralities)
    
    # Spearman's correlation using only the co-citation network of 92-02
    print("Processing the Spearman's correlation.")
    files = ['all_degree_cen_cocitation-92-02.txt', 'all_betweenness_cen_cocitation-92-02.txt',
    'all_closeness_cen_cocitation-92-02.txt', 'all_page_rank_cocitation-92-02.txt']
    centralities = []
    centrality_rankings = []
    for file in files:
        centrality_dict = {}
        ranking = {}
        with open(path + file, 'r') as f:
            for line in f:
                nodeid, value, rank = line.split()
                centrality_dict[nodeid] = float(value)
                ranking[nodeid] = int(rank)
        centralities.append(centrality_dict)
        centrality_rankings.append(ranking)
        
    degree_cen, betweenness_cen, closeness_cen, page_rank = centralities
    degree_ranking, betweenness_ranking, closeness_ranking, page_rank_ranking = centrality_rankings
    citation, degree, betweenness, closeness, pagerank, nodes = [], [], [], [], [], []
    for nodeid, count in sorted(citation_count.items(), key=lambda x : x[1], reverse=True):
        if nodeid in degree_cen and nodeid in betweenness_cen and nodeid in closeness_cen and nodeid in page_rank:
            citation.append(count)
            degree.append(degree_cen[nodeid])
            betweenness.append(betweenness_cen[nodeid])
            closeness.append(closeness_cen[nodeid])
            pagerank.append(page_rank[nodeid])
            nodes.append(nodeid)
    array1 = [citation, degree, betweenness, closeness, pagerank]
    array2 = array1[:]
    names = ["Citation", "Degree", "Betweenness", "Closeness", "PageRank"]
    centrality_rankings = [degree_ranking, betweenness_ranking, closeness_ranking, page_rank_ranking]
    get_spearman_correlation(array1, array2, names, citation_count_ranking, centrality_rankings)
    
    # Small world analysis
    print("Determining small world phenomenon.")
    file = "cocitation-92-94.txt"
    cocitation_net = nx.read_edgelist(file_name, nodetype=int)
    small_world_analysis(cocitation_net)
    
if __name__ == "__main__":
    main()